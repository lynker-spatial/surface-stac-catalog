# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "polars",
#     "pystac",
#     "rasterio",
#     "rioxarray",
#     "xarray",
#     "netcdf4",
#     "shapely",
#     "pydap"
# ]
# ///

#!/usr/bin/env python3
# scripts/generate_catalog.py
# Build a STAC catalog from data/metadata.csv:
# root -> domain collection -> per-row sub-collection -> one item per asset URL
# Also exports derived bounding boxes to dist/derived_bboxes.geojson for QGIS.

from __future__ import annotations

import json
import os
import sys
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
from urllib.parse import urlparse
import hashlib

import polars as pl
import pystac
from pystac.extensions.projection import ProjectionExtension

# ----------------------------
# Config
# ----------------------------
INPUT_CSV = Path("data/metadata.csv")
OUT_DIR = Path("dist")

# Required CSV columns (bbox/datetime optional)
REQUIRED_COLS = {
    "id",
    "domain",
    "region",
    "release_version",
    "source",
    "resolution",
    "horizontal_crs",
    "original_vdatum",
    "current_vdatum",
    "vertical_datum_conversion",
    "priority",
    "source_url",
    "asset_urls",
}

# Fallback datetime if nothing is provided or derivable (override via env)
FALLBACK_DT_STR = os.environ.get("STAC_FALLBACK_DATETIME", "1970-01-01T00:00:00Z")

# Collect unreachable/bad URLs
UNREACHABLE_URLS: List[str] = []


# ----------------------------
# Env helpers (GDAL/PROJ on Windows/conda)
# ----------------------------
def ensure_gdal_env():
    """Set GDAL/PROJ paths on Windows if not already defined"""
    if sys.platform != "win32":
        return
    
    prefix = os.environ.get("CONDA_PREFIX") or os.environ.get("VIRTUAL_ENV") or sys.prefix
    if prefix:
        conda_gdal = os.path.join(prefix, "Library", "share", "gdal")
        conda_proj = os.path.join(prefix, "Library", "share", "proj")
    
        if os.path.exists(conda_gdal):
            os.environ.setdefault("GDAL_DATA", conda_gdal)
        if os.path.exists(conda_proj):
            os.environ.setdefault("PROJ_LIB", conda_proj)

ensure_gdal_env()

# ----------------------------
# Dataclass (bbox/datetime optional)
# ----------------------------
@dataclass
class STACMetadata:
    id: str
    domain: str
    region: str
    source: str
    release_version: str
    resolution: str
    horizontal_crs: str
    original_vdatum: str
    current_vdatum: str
    vertical_datum_conversion: str
    priority: Optional[int]
    source_url: str
    asset_urls: str
    bbox: Optional[str] = None
    datetime: Optional[str] = None


# ----------------------------
# Utilities
# ----------------------------
def sanitize_id(value) -> str:
    """Return a STAC id (string, limited charset)."""
    s = str(value).strip()
    if not s:
        s = "unnamed"
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)  # keep alnum, dot, underscore, dash
    return s


def read_metadata_from_csv() -> pl.DataFrame:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"'{INPUT_CSV}' does not exist. Expected at project_root/data/metadata.csv"
        )

    df = pl.read_csv(
        INPUT_CSV,
        try_parse_dates=False,
        infer_schema_length=10000,
        null_values=["", "NA", "N/A", "Null", "null"],
        ignore_errors=False,
    )

    # Normalize header names (trim spaces)
    df = df.rename({c: c.strip() for c in df.columns})

    # Validate required columns
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV missing required columns: {sorted(missing)}. Present: {df.columns}"
        )

    # Safe cast priority by name (non-numeric -> null)
    df = df.with_columns(pl.col("priority").cast(pl.Int64, strict=False))

    # Ensure optional columns exist
    for opt in ("bbox", "datetime"):
        if opt not in df.columns:
            df = df.with_columns(pl.lit(None).alias(opt))

    # Force id/domain to strings to avoid PySTAC cache key errors
    df = df.with_columns(
        [
            pl.col("id").cast(pl.Utf8),
            pl.col("domain").cast(pl.Utf8),
        ]
    )

    return df


def try_parse_bbox(bbox_str: Optional[str]) -> Optional[List[float]]:
    if not bbox_str:
        return None
    v = bbox_str.strip()
    try:
        arr = json.loads(v) if v.startswith("[") else [float(x) for x in v.split(",")]
        if len(arr) == 4 and all(isinstance(a, (int, float)) for a in arr):
            return [float(a) for a in arr]
    except Exception:
        pass
    return None


def parse_iso_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    v = s.strip()
    try:
        if v.endswith("Z"):
            return datetime.fromisoformat(v.replace("Z", "+00:00"))
        dt = datetime.fromisoformat(v)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


# Fast reachability with immediate logging
def is_url_ok(url: str, timeout: int = 3) -> bool:
    """Fast reachability check; logs failures immediately.
    Bypasses private S3 URLs to prevent skipping valid assets"""

    hl = url.lower()

    # trust private S3 bucket paths / endpoints
    if "spatial-water-noaa" in hl:
        return True

    # standard check for public assets
    try:
        req = Request(url, method="HEAD")
        with urlopen(req, timeout=timeout) as resp:
            return 200 <= resp.status < 400
    except Exception:
        try:
            req = Request(url)  # GET fallback
            with urlopen(req, timeout=timeout) as resp:
                return 200 <= resp.status < 400
        except (HTTPError, URLError) as e:
            msg = getattr(e, "reason", getattr(e, "code", str(e)))
            print(f"[unreachable] {url}  ({msg})")
            UNREACHABLE_URLS.append(url)
            return False
        except Exception as e:
            print(f"[unreachable] {url}  ({e})")
            UNREACHABLE_URLS.append(url)
            return False


# matches http/https URLs until whitespace or a closing bracket/quote
_URL_RE = re.compile(r'https?://[^\s\'",]+', re.IGNORECASE)


def parse_asset_urls(value: Optional[str]) -> List[str]:
    """
    Robustly extract all URLs from the cell. Handles:
      - JSON lists (single or double quotes)
      - plain text with commas/semicolons/newlines
      - cells that mistakenly contain *multiple* URLs inside one quoted string
    """
    s = (value or "").strip()
    if not s:
        return []

    out: List[str] = []

    # 1) Try JSON (allow single quotes); if it parses, collect strings inside
    try:
        as_json = s if '"' in s else s.replace("'", '"')
        data = json.loads(as_json)
        if isinstance(data, str):
            out.extend(_URL_RE.findall(data))
        elif isinstance(data, list):
            for x in data:
                out.extend(_URL_RE.findall(x if isinstance(x, str) else str(x)))
        else:
            out.extend(_URL_RE.findall(s))
    except Exception:
        # 2) Not JSON -> just regex over the raw cell
        out.extend(_URL_RE.findall(s))

    # 3) Normalize & dedupe
    cleaned = []
    seen = set()
    for u in out:
        u2 = u.strip().rstrip(",")  # trim trailing comma
        if u2 and u2 not in seen:
            seen.add(u2)
            cleaned.append(u2)
    return cleaned


def url_candidates(u: str) -> List[str]:
    """Return candidate URLs to try for a given href (e.g., .tiff -> .tif)."""
    out = [u]
    if u.lower().endswith(".tiff"):
        out.append(u[:-5] + ".tif")
    return out


def epsg_from_crs(crs_str: Optional[str]) -> Optional[int]:
    if not crs_str:
        return None
    m = re.search(r"epsg:(\d+)", crs_str, flags=re.IGNORECASE)
    return int(m.group(1)) if m else None


def geojson_polygon_from_bbox(bbox: List[float]) -> dict:
    """Return GeoJSON Polygon from [minx, miny, maxx, maxy]."""
    minx, miny, maxx, maxy = bbox
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [minx, miny],
                [minx, maxy],
                [maxx, maxy],
                [maxx, miny],
                [minx, miny],
            ]
        ],
    }


# ----------------------------
# BBOX/CRS from a single asset (WGS84 bbox)
# ----------------------------
def _transform_to_wgs84(bounds, src_crs) -> Optional[List[float]]:
    """Transform (minx, miny, maxx, maxy) from src_crs to EPSG:4326."""
    try:
        from rasterio.warp import transform_bounds

        return list(transform_bounds(src_crs, "EPSG:4326", *bounds, densify_pts=21))
    except Exception as e:
        print(f"[warn] transform to WGS84 failed: {e}")
        return None


def _bbox_epsg_from_geotiff_like(
    url: str,
) -> Optional[Tuple[List[float], Optional[int]]]:
    """Handle .tif/.tiff, .zip and .vrt via rasterio and active AWS credentials"""
    try:
        import rasterio

        # Translate S3 HTTPS endpoints to GDAL /vsis3/
        s3_prefix = "https://spatial-water-noaa.s3.us-east-1.amazonaws.com/"
        gdal_url = (
            url.replace(s3_prefix, "/vsis3/spatial-water-noaa/")
            if url.startswith(s3_prefix)
            else url
        )
        env_opts = {
            "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",  # speed over HTTP
            "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff,.vrt,.xml,.zip",
        }

        # Check if AWS crendentials or profiles are active in the environment
        if os.environ.get("AWS_ACCESS_KEY_ID") or os.environ.get("AWS_PROFILE"):
            env_opts["AWS_NO_SIGN_REQUEST"] = "NO"
            if os.environ.get("AWS_PROFILE"):
                env_opts["AWS_PROFILE"] = os.environ.get("AWS_PROFILE")
                env_opts["AWS_SDK_LOAD_CONFIG"] = "1"
            if os.environ.get("AWS_DEFAULT_REGION"):
                env_opts["AWS_DEFAULT_REGION"] = os.environ.get("AWS_DEFAULT_REGION")
        else:
            # Fallback for public S3 buckets when no AWS keys are present
            env_opts["AWS_NO_SIGN_REQUEST"] = "YES"

        with rasterio.Env(**env_opts):
            with rasterio.open(gdal_url) as ds:
                b = ds.bounds  # (left, bottom, right, top) in ds.crs
                crs = ds.crs
                epsg = crs.to_epsg() if crs else None
                if crs:
                    b_wgs84 = _transform_to_wgs84(
                        (b.left, b.bottom, b.right, b.top), crs
                    )
                else:
                    b_wgs84 = None
                return (b_wgs84, epsg) if b_wgs84 else None

    except Exception as e:
        print(f"[warn] GeoTIFF/VRT read failed for {url}: {e}")
        UNREACHABLE_URLS.append(url)
        return None


def _bbox_epsg_from_netcdf(url: str) -> Optional[Tuple[List[float], Optional[int]]]:
    """Try rioxarray/xarray for CF NetCDF (local or THREDDS/OPeNDAP). Returns (bbox_wgs84, epsg)."""
    try:
        import rioxarray as rxr
        import xarray as xr
        import rasterio

    except Exception as e:
        print(f"[warn] Failed to import xarray/rioxarray: {e}")
        return None

    # xarray (more reliable for OPeNDAP/dodsC)
    # We explicitly use the 'netcdf4' engine and 'chunks={}' to ensure no data is downloaded.
    try:
        # Use chunks={} to force lazy loading with Dask
        with xr.open_dataset(url, engine="netcdf4", chunks={}) as ds:
            for _, da in ds.data_vars.items():
                dims = set(da.dims)
                if {"y", "x"}.issubset(dims) or {"lat", "lon"}.issubset(dims):
                    try:
                        if {"lat", "lon"}.issubset(da.coords):
                            lat = da["lat"].values
                            lon = da["lon"].values
                            lon_min = float(lon.min())
                            lon_max = float(lon.max())
                            lat_min = float(lat.min())
                            lat_max = float(lat.max())
                            return [lon_min, lat_min, lon_max, lat_max], 4326
                    except Exception:
                        pass
                    try:
                        _ = da.rio.crs  # may raise
                        bounds = da.rio.bounds()
                        crs = da.rio.crs
                        if crs and bounds:
                            epsg = crs.to_epsg() if crs else None
                            if epsg == 4326:
                                return [
                                    bounds[0],
                                    bounds[1],
                                    bounds[2],
                                    bounds[3],
                                ], epsg
                            b_wgs84 = _transform_to_wgs84(
                                (bounds[0], bounds[1], bounds[2], bounds[3]), crs
                            )
                            if b_wgs84:
                                return b_wgs84, epsg
                    except Exception:
                        pass

            # CF global attributes fallback
            for names in (
                (
                    "geospatial_lon_min",
                    "geospatial_lat_min",
                    "geospatial_lon_max",
                    "geospatial_lat_max",
                ),
                (
                    "geospatial_longitude_min",
                    "geospatial_latitude_min",
                    "geospatial_longitude_max",
                    "geospatial_latitude_max",
                ),
            ):
                try:
                    lon_min = float(ds.attrs.get(names[0]))
                    lat_min = float(ds.attrs.get(names[1]))
                    lon_max = float(ds.attrs.get(names[2]))
                    lat_max = float(ds.attrs.get(names[3]))
                    return [lon_min, lat_min, lon_max, lat_max], 4326
                except Exception:
                    continue
    except Exception:
        pass  # Failed, so we'll try rioxarray

    # rioxarray
    try:
        env_opts = {
            "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
            "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".nc",
        }
        with rasterio.Env(**env_opts):
            da = rxr.open_rasterio(url, masked=True)
            crs = getattr(da.rio, "crs", None)
            bounds = getattr(da.rio, "bounds", None)
            if crs and bounds:
                epsg = crs.to_epsg() if crs else None
                if epsg == 4326:
                    return [bounds[0], bounds[1], bounds[2], bounds[3]], epsg
                b_src = (bounds[0], bounds[1], bounds[2], bounds[3])
                b_wgs84 = _transform_to_wgs84(b_src, crs)
                if b_wgs84:
                    return b_wgs84, epsg
    except Exception:
        pass  # Both strategies failed

    # If we're here, both failed
    print(f"[warn] NetCDF read failed for {url}")
    UNREACHABLE_URLS.append(url)
    return None


def bbox_from_single_asset(
    asset_url: str,
) -> Tuple[Optional[List[float]], Optional[int]]:
    """Return (bbox_WGS84, epsg) for a single asset URL."""
    hl = asset_url.lower()
    if hl.endswith((".tif", ".tiff", ".vrt")):
        got = _bbox_epsg_from_geotiff_like(asset_url)
    elif hl.endswith(".nc") or "thredds/dodsC" in hl or "opendap" in hl:
        got = _bbox_epsg_from_netcdf(asset_url)
    else:
        got = None
    if not got:
        return None, None
    return got


# ----------------------------
# Time handling: datetime OR start/end
# ----------------------------
def build_item_time(
    csv_datetime: Optional[str], asset_url: Optional[str]
) -> Tuple[Optional[datetime], dict]:
    """
    Returns (item_datetime, properties). If csv_datetime is missing:
    - Try (start,end) from NetCDF asset; if found, set start/end and return (None, props)
    - Else, return (fallback_datetime, props)
    """
    props: dict = {}

    # 1) direct datetime from CSV
    dt = parse_iso_dt(csv_datetime) if csv_datetime else None
    if dt is not None:
        return dt, props

    # 2) derive from a single NetCDF asset if available
    if asset_url:
        rng = temporal_from_single_asset(asset_url)
        if rng:
            start, end = rng
            props["start_datetime"] = start.isoformat().replace("+00:00", "Z")
            props["end_datetime"] = end.isoformat().replace("+00:00", "Z")
            return None, props

    # 3) fallback single datetime
    fb = parse_iso_dt(FALLBACK_DT_STR) or datetime(1970, 1, 1, tzinfo=timezone.utc)
    return fb, props


def temporal_from_single_asset(asset_url: str) -> Optional[Tuple[datetime, datetime]]:
    """Return (start,end) from a NetCDF asset using 'time' coordinate, if available."""
    try:
        import xarray as xr
        import numpy as np
    except Exception:
        return None

    if not (
        asset_url.lower().endswith(".nc")
        or "thredds/dodsC" in asset_url.lower()
        or "opendap" in asset_url.lower()
    ):
        return None

    def to_dt(v) -> Optional[datetime]:
        try:
            if isinstance(v, np.datetime64):
                ts_ns = v.astype("datetime64[ns]").astype("int64")
                return datetime.utcfromtimestamp(ts_ns / 1e9).replace(
                    tzinfo=timezone.utc
                )
        except Exception:
            pass
        if isinstance(v, (str, bytes)):
            return parse_iso_dt(v)
        return None

    try:
        ds = xr.open_dataset(asset_url)
        for name in ("time", "t"):
            if name in ds.coords or name in ds.variables:
                t = ds[name].values
                try:
                    tmin, tmax = t.min(), t.max()
                except Exception:
                    continue
                dt_start, dt_end = to_dt(tmin), to_dt(tmax)
                if dt_start and dt_end:
                    return dt_start, dt_end
    except Exception:
        return None
    return None


# ----------------------------
# Build STAC (domain collection -> per-row sub-collection -> items)
# ----------------------------
def main() -> None:
    ensure_gdal_env()

    print(f"[info] Reading: {INPUT_CSV.resolve()}")
    df = read_metadata_from_csv()

    root = pystac.Catalog(
        id="hydrofabric-topobathy-data",
        title="Hydrofabric Master Surface DEM Data Catalog",
        description="A STAC catalog of topobathymetric digital elevation models (DEMs) data sources for CONUS and oCONUS domains supporting Office of Water Prediction (OWP) operations, including National Water Model (NWM) and the Flood Inundation Mapping (FIM) Program",
    )

    domain_collections: Dict[str, pystac.Collection] = {}
    sub_collections: Dict[str, pystac.Collection] = {}
    derived_bboxes: List[Dict] = []  # collect for QGIS GeoJSON export

    for row in df.iter_rows(named=True):
        meta = STACMetadata(**{k: row.get(k) for k in row})

        domain_id = sanitize_id(meta.domain)
        row_id = sanitize_id(meta.id)  # CSV "id" becomes sub-collection id base
        region_id = sanitize_id(meta.region)


        domain_names = {
            "alaska": "Alaska", 
            "conus-atlantic-gulf": "CONUS Atlantic-Gulf", 
            "conus-pacific": "CONUS Pacific",
            "great-lakes": "Great Lakes",
            "hawaii": "Hawaii",
            "puerto-rico-virgin-islands": "Puerto Rico & US Virgin Islands",
        }
        
        # Get or create domain collection
        domain_coll = domain_collections.get(domain_id)
        if domain_coll is None:
            
            clean_title = domain_names.get(domain_id, domain_id.replace("-", " ").title())
            
            domain_coll = pystac.Collection(
                id=domain_id,
                title=clean_title,
                description=f"Topobathymetric surface elevation data sources for the {clean_title} region.",
                extent=pystac.Extent(
                    spatial=pystac.SpatialExtent([[-180.0, -90.0, 180.0, 90.0]]),
                    temporal=pystac.TemporalExtent([[None, None]]),
                ),
                license="proprietary",
                keywords=[
                    "topobathy",
                    "hydrofabric",
                    "DEM",
                    "coastal",
                    "NWM",
                    "oCONUS",
                    "CONUS",
                ],
                providers=[
                    pystac.Provider(
                        name="NOAA-OWP",
                        roles=[pystac.ProviderRole.PRODUCER, pystac.ProviderRole.HOST],
                        # url="",
                    )
                ],
            )
            domain_collections[domain_id] = domain_coll
            root.add_child(domain_coll)

        # Create REGION sub-collection under this domain
        sub_id = f"{domain_id}-{region_id}"  # e.g., "conus-pacific-coos-bay-oregon"
        sub_coll = sub_collections.get(sub_id)

        if sub_coll is None:
            sub_title = f"{meta.region} ({meta.domain})"
            sub_desc = f"Topobathymetric DEM dataset collection for {meta.region} within the {meta.domain} domain."

            sub_coll = pystac.Collection(
                id=sub_id,
                title=sub_title,
                description=sub_desc,
                extent=pystac.Extent(
                    spatial=pystac.SpatialExtent([[-180.0, -90.0, 180.0, 90.0]]),
                    temporal=pystac.TemporalExtent([[None, None]]),
                ),
                license="proprietary",
            )
            sub_collections[sub_id] = sub_coll
            domain_coll.add_child(sub_coll)

        # Parse all asset URLs for this row
        raw_urls = parse_asset_urls(meta.asset_urls)
        print(f"[row] {row_id} — {len(raw_urls)} asset(s) found")

        # Reachable candidates with simple (.tiff -> .tif) fallback
        reachable_urls: List[str] = []
        for u in raw_urls:
            chosen = None
            for c in url_candidates(u):
                if is_url_ok(c):
                    chosen = c
                    break
            if chosen:
                reachable_urls.append(chosen)
            # is_url_ok logs unreachable immediately

        if not reachable_urls:
            print(f"[skip-row] {row_id}: no reachable assets")
            continue

        # For each reachable asset, make a separate Item inside the sub-collection
        for idx, href in enumerate(reachable_urls, start=1):
            # Per-asset ID: <row_id>_<filename_stem>_<short_hash>
            parsed = urlparse(href)
            fname = os.path.basename(parsed.path) or f"asset_{idx}"
            stem = os.path.splitext(fname)[0] or f"asset_{idx}"
            short_hash = hashlib.sha1(href.encode("utf-8")).hexdigest()[:8]
            item_id = sanitize_id(f"{row_id}_{stem}_{short_hash}")

            # Per-asset bbox/epsg
            item_bbox, epsg_from_asset = bbox_from_single_asset(href)

            # Time: CSV datetime or derive from this asset (NetCDF) or fallback
            item_datetime, time_props = build_item_time(meta.datetime, href)

            # Build item properties
            props = {
                **time_props,
                "domain": meta.domain,
                "region": meta.region,
                "release_version": meta.release_version,
                "source": meta.source,
                "resolution": meta.resolution,
                "horizontal_crs": meta.horizontal_crs,
                "original_vdatum": meta.original_vdatum,
                "current_vdatum": meta.current_vdatum,
                "vertical_datum_conversion": meta.vertical_datum_conversion,
                "priority": meta.priority,
                "source_url": meta.source_url,
                "filename": fname,
            }
            props = {k: v for k, v in props.items() if v not in (None, "")}

            # Create item (STAC wants WGS84 bbox/geometry; allow None)
            item = pystac.Item(
                id=item_id,
                geometry=geojson_polygon_from_bbox(item_bbox) if item_bbox else None,
                bbox=item_bbox if item_bbox else None,
                datetime=item_datetime,  # None allowed iff start/end in props
                properties=props,
                collection=sub_coll,
            )

            # Projection extension
            epsg = epsg_from_crs(meta.horizontal_crs) or epsg_from_asset
            if epsg:
                ProjectionExtension.ext(item, add_if_missing=True).epsg = epsg

            # Single asset on the item
            media_type = None
            hl = href.lower()
            if hl.endswith((".tif", ".tiff")):
                media_type = (
                    pystac.MediaType.COG if "cog" in hl else pystac.MediaType.GEOTIFF
                )
            elif hl.endswith(".vrt"):
                media_type = "application/x-vrt"
            elif hl.endswith(".nc") or "thredds/dodsC" in hl or "opendap" in hl:
                media_type = "application/x-netcdf"
            elif hl.endswith(".csv"):
                media_type = pystac.MediaType.TEXT

            item.add_asset(
                key="data",
                asset=pystac.Asset(
                    href=href, media_type=media_type, roles=["data"], title=f"{fname}"
                ),
            )

            # Optional source link
            if meta.source_url:
                item.add_link(pystac.Link(rel="via", target=meta.source_url))

            # Add to sub-collection
            sub_coll.add_item(item)

            # Collect for GeoJSON export
            if item_bbox:
                derived_bboxes.append(
                    {
                        "id": item.id,
                        "domain": meta.domain,
                        "row_id": row_id,
                        "region": meta.region,
                        "source": meta.source,
                        "resolution": meta.resolution,
                        "href": href,
                        "bbox": item_bbox,  # [minx, miny, maxx, maxy] in EPSG:4326
                    }
                )

        # Tighten extents on sub and domain
        sub_coll.update_extent_from_items()
        domain_coll.update_extent_from_items()

    # Save catalog
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    root.normalize_hrefs(str(OUT_DIR))
    root.save(pystac.CatalogType.SELF_CONTAINED)
    print(f"[ok] STAC catalog written to: {OUT_DIR.resolve()}")

    # Final report of unreachable URLs
    if UNREACHABLE_URLS:
        deduped = sorted(set(UNREACHABLE_URLS))
        print("\n[report] Unreachable asset URLs:")
        for u in deduped:
            print(" -", u)
        report_path = OUT_DIR / "unreachable_urls.txt"
        with report_path.open("w", encoding="utf-8") as f:
            for u in deduped:
                f.write(u + "\n")
        print(f"[report] wrote {len(deduped)} unreachable URL(s) -> {report_path}")
    else:
        print("\n[report] All asset URLs reachable")

    # --- Export derived bboxes as GeoJSON for QGIS/geojson.io ---
    if derived_bboxes:
        features = []
        for row in derived_bboxes:
            minx, miny, maxx, maxy = row["bbox"]
            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [minx, miny],
                                [minx, maxy],
                                [maxx, maxy],
                                [maxx, miny],
                                [minx, miny],
                            ]
                        ],
                    },
                    "properties": {k: v for k, v in row.items() if k != "bbox"},
                }
            )
        gj = {"type": "FeatureCollection", "features": features}
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        gj_path = OUT_DIR / "derived_bboxes.geojson"
        with gj_path.open("w", encoding="utf-8") as f:
            json.dump(gj, f, indent=2)
        print(f"[ok] wrote {len(features)} bbox feature(s) -> {gj_path}")
    else:
        print("[info] no bboxes derived; skipped GeoJSON export")


if __name__ == "__main__":
    main()

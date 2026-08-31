import logging, os, urllib.request
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np, pandas as pd

logger = logging.getLogger(__name__)
SST_EP = os.getenv("OCEAN_SST_OPENDAP", "https://coastwatch.pfeg.noaa.gov/erddap/griddap/jplMURSST41")
SSH_EP = os.getenv("OCEAN_SSH_OPENDAP", "https://coastwatch.pfeg.noaa.gov/erddap/griddap/nesdisSSH1day")
SSS_EP = os.getenv("OCEAN_SSS_OPENDAP", "https://coastwatch.pfeg.noaa.gov/erddap/griddap/coastwatchSMOSv662SSS3day")

def haversine_km(lat1, lon1, lat2, lon2):
    from math import radians, sin, cos, atan2, sqrt
    R = 6371.0; dlat, dlon = radians(lat2-lat1), radians(lon2-lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return R*2*atan2(sqrt(a), sqrt(1-a))

def _q(url, var, lat, lon, dt, dlat=0.25, dlon=0.25, tw=24.0, alt=False, timeout=15):
    ts = (dt-timedelta(hours=tw)).strftime("%Y-%m-%dT%H:%M:%SZ")
    te = (dt+timedelta(hours=tw)).strftime("%Y-%m-%dT%H:%M:%SZ")
    la1,la2 = round(lat-dlat,3),round(lat+dlat,3)
    lo1,lo2 = round(lon-dlon,3),round(lon+dlon,3)
    q = f"{var}[({ts}):1:({te})]" + ("[(0.0):1:(0.0)]" if alt else "") + f"[({la1}):1:({la2})][({lo1}):1:({lo2})]"
    req = urllib.request.Request(f"{url}.csv?{q}", headers={"User-Agent":"OceanEmbed/2.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            df = pd.read_csv(r, skiprows=[1])
            if var in df.columns: df = df.dropna(subset=[var])
            return df if len(df) > 0 else None
    except: return None

def fetch_nearest_surface(lat, lon, dt=None, tw=24.0, max_km=25.0, dataset_url=None):
    if dt is None: return None
    tdt = pd.to_datetime(dt).to_pydatetime() if not isinstance(dt, datetime) else dt
    if tdt.tzinfo: tdt = tdt.replace(tzinfo=None)
    out = dict(sst=None,ssh=None,sss=None,obs_lat=None,obs_lon=None,obs_time=None,distance_km=None,sst_source=None,ssh_source=None,sss_source=None)
    df = _q(dataset_url or SST_EP, "analysed_sst", lat, lon, tdt, 0.15, 0.15, tw, timeout=20)
    if df is not None:
        df["dk"]=[haversine_km(lat,lon,r["latitude"],r["longitude"]) for _,r in df.iterrows()]
        df["td"]=[abs((pd.to_datetime(t).tz_localize(None)-tdt).total_seconds())/3600 for t in df["time"]]
        v=df[(df["dk"]<=max_km)&(df["td"]<=tw)]
        if len(v)>0:
            b=v.loc[(v["dk"]+v["td"]*0.5).idxmin()]
            out.update(sst=float(b["analysed_sst"]),obs_lat=float(b["latitude"]),obs_lon=float(b["longitude"]),obs_time=pd.to_datetime(b["time"]),distance_km=float(b["dk"]),sst_source="JPL_MUR_SST_v4.1")
    df2=_q(SSH_EP,"sla",lat,lon,tdt,0.35,0.35,tw,timeout=20)
    if df2 is not None:
        df2["dk"]=[haversine_km(lat,lon,r["latitude"],r["longitude"]) for _,r in df2.iterrows()]
        df2["td"]=[abs((pd.to_datetime(t).tz_localize(None)-tdt).total_seconds())/3600 for t in df2["time"]]
        v=df2[(df2["dk"]<=max_km)&(df2["td"]<=tw)]
        if len(v)>0:
            b=v.loc[(v["dk"]+v["td"]*0.5).idxmin()]; out["ssh"]=float(b["sla"]); out["ssh_source"]="NOAA_NESDIS_SLA"
            if out["obs_lat"] is None: out.update(obs_lat=float(b["latitude"]),obs_lon=float(b["longitude"]),obs_time=pd.to_datetime(b["time"]),distance_km=float(b["dk"]))
    df3=_q(SSS_EP,"sss",lat,lon,tdt,0.35,0.35,tw*2,alt=True,timeout=20)
    if df3 is not None:
        df3["dk"]=[haversine_km(lat,lon,r["latitude"],r["longitude"]) for _,r in df3.iterrows()]
        df3["td"]=[abs((pd.to_datetime(t).tz_localize(None)-tdt).total_seconds())/3600 for t in df3["time"]]
        v=df3[(df3["dk"]<=max_km)&(df3["td"]<=tw*2)]
        if len(v)>0: b=v.loc[(v["dk"]+v["td"]*0.5).idxmin()]; out["sss"]=float(b["sss"]); out["sss_source"]="ESA_SMOS_L3_SSS"
    return out if out["sst"] is not None else None

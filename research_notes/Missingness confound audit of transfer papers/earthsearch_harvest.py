"""Harvest Sentinel-2 L1C granule-level eo:cloud_cover from AWS Earth Search STAC
for representative cropland points, 2018-01-01..2024-12-31, and summarise by month."""
import requests, json, time, csv, sys, statistics as st
from collections import defaultdict
API="https://earth-search.aws.element84.com/v1/search"
REGIONS={  # name: (lat, lon)
 "Brittany_FR_Rennes":(48.1,-1.7),"N_France_Somme":(49.9,2.3),"Denmark_Jutland_Viborg":(56.4,9.4),
 "Austria_Marchfeld":(48.25,16.75),"Bavaria_Straubing":(48.8,12.6),"Brandenburg_Prenzlau":(53.2,13.8),
 "Catalonia_Lleida":(41.6,0.7),"Netherlands_Flevoland":(52.5,5.5),"Belgium_Hesbaye":(50.6,4.7),
 "WesternCape_Overberg_Caledon":(-34.2,19.5),"WesternCape_Swartland_Malmesbury":(-33.4,18.7),
 "Kenya_Bungoma":(0.6,34.6),"Kenya_Nakuru":(-0.4,36.0),"Tanzania_Mbeya":(-8.9,33.5),
 "Nigeria_Kaduna":(10.5,7.4),"Ghana_Tamale":(9.4,-0.85),"Ethiopia_Arsi":(7.9,39.1),
 "Mali_Segou":(13.4,-6.2),"Senegal_Kaolack":(14.15,-16.1),
 "India_Punjab_Ludhiana":(30.9,75.85),"India_AndhraPradesh_Guntur":(16.3,80.4),
 "NEChina_Songnen_Harbin":(46.0,125.5),"NorthChinaPlain_Hengshui":(37.7,115.7),
 "Yunnan_Qujing":(25.5,103.8),"Sichuan_Chengdu":(30.6,104.0),"Iran_Fars_Marvdasht":(29.9,52.8),
 "US_CornBelt_Ames":(42.0,-93.6),"US_CentralValley_Fresno":(36.6,-119.8),
 "Argentina_Pergamino":(-33.9,-60.6),"Brazil_MatoGrosso_Sorriso":(-12.5,-55.7),"Brazil_RioGrandeSul_CruzAlta":(-28.6,-53.6),
 "Australia_Victoria_Horsham":(-36.7,142.2),"Australia_WA_Merredin":(-31.5,118.3)}
def harvest(lat,lon,start="2018-01-01T00:00:00Z",end="2024-12-31T23:59:59Z",coll="sentinel-2-l1c",d=0.02):
    body={"collections":[coll],"bbox":[lon-d,lat-d,lon+d,lat+d],"datetime":f"{start}/{end}","limit":500,
          "fields":{"include":["id","properties.datetime","properties.eo:cloud_cover","properties.grid:code"],"exclude":["geometry","assets","links","bbox"]}}
    items=[]; nxt=body
    while True:
        for attempt in range(4):
            r=requests.post(API,json=nxt,timeout=90)
            if r.status_code==200: break
            time.sleep(3*(attempt+1))
        r.raise_for_status(); j=r.json(); items+=j["features"]
        link=[l for l in j.get("links",[]) if l.get("rel")=="next"]
        if not link: break
        nxt=link[0].get("body") or {**body,**{k:v for k,v in link[0].items() if k=="body"}}
        if "body" in link[0]: nxt=link[0]["body"]
        else: break
        time.sleep(0.3)
    return items
out=open(sys.argv[1],"w",newline=""); w=csv.writer(out)
w.writerow(["region","lat","lon","month","n_granules","median_cc","mean_cc","frac_cc_lt20","frac_cc_lt50","n_years"])
raw=open(sys.argv[1].replace(".csv","_raw.jsonl"),"w")
for name,(lat,lon) in REGIONS.items():
    try: items=harvest(lat,lon)
    except Exception as e: print(name,"ERR",e,file=sys.stderr); continue
    bym=defaultdict(list); yrs=set()
    for it in items:
        p=it["properties"]; cc=p.get("eo:cloud_cover")
        if cc is None: continue
        m=int(p["datetime"][5:7]); yrs.add(p["datetime"][:4]); bym[m].append(cc)
        raw.write(json.dumps({"region":name,"id":it["id"],"dt":p["datetime"],"cc":cc,"tile":p.get("grid:code")})+"\n")
    for m in range(1,13):
        v=bym.get(m,[])
        if v: w.writerow([name,lat,lon,m,len(v),round(st.median(v),1),round(st.mean(v),1),round(sum(x<20 for x in v)/len(v),2),round(sum(x<50 for x in v)/len(v),2),len(yrs)])
        else: w.writerow([name,lat,lon,m,0,"","","","",len(yrs)])
    out.flush(); print(name,len(items),"granules",file=sys.stderr)

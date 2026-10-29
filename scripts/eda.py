"""Profile current extracted label CSVs; originals are read-only."""
from pathlib import Path
import json, re, unicodedata
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs'/'eda'; OUT.mkdir(parents=True,exist_ok=True)
profiles=[]; tables={};inventory=[]
for path in sorted((ROOT/'data').rglob('*.csv')):
 rel=unicodedata.normalize('NFC',str(path.relative_to(ROOT/'data')))
 inventory.append({'path':rel,'bytes':path.stat().st_size})
 parts=path.parts
 if path.parent.name not in ['TL_csv','VL_csv'] or 'poi_master' in path.name:continue
 year='2023' if '2023' in rel else '2022' if '2022' in rel else None
 if year is None:continue
 split='train' if path.parent.name=='TL_csv' else 'validation'
 df=pd.read_csv(path,dtype=str,keep_default_na=False).replace(r'^\s*$',pd.NA,regex=True)
 key=path.name.split('_')[0:3];key='_'.join(key)
 # Prefix-based table selection avoids Korean filename normalization differences.
 tables[(year,split,path.name)]=df
 for col in df:
  ss=df[col]; nn=pd.to_numeric(ss,errors='coerce'); n=ss.notna().sum()
  profiles.append({'year':year,'split':split,'table':path.name,'rows':len(df),'columns':len(df.columns),'duplicate_rows':int(df.duplicated().sum()),'column':col,'inferred_type':'numeric-like' if n and nn.notna().sum()==n else 'string','null_count':int(ss.isna().sum()),'null_pct':float(ss.isna().mean()*100),'unique':int(ss.nunique())})
pd.DataFrame(inventory).to_csv(OUT/'csv_inventory.csv',index=False)
pd.DataFrame(profiles).to_csv(OUT/'column_profiles.csv',index=False)
def load(year,prefix):
 ds=[v for (y,s,n),v in tables.items() if y==year and n.startswith(prefix)]
 if not ds:raise RuntimeError('Missing label table '+year+' '+prefix)
 return pd.concat(ds,ignore_index=True).drop_duplicates()
allv=[];metrics={}
for year in ['2022','2023']:
 travel=load(year,'tn_travel_');users=load(year,'tn_traveller_master')
 if travel.TRAVEL_ID.duplicated().any() or users.TRAVELER_ID.duplicated().any():raise ValueError('Ambiguous parent keys')
 for (y,split,n),visits in tables.items():
  if y!=year or not n.startswith('tn_visit_area_info'):continue
  v=visits.merge(travel[['TRAVEL_ID','TRAVELER_ID']],how='left',on='TRAVEL_ID',validate='many_to_one')
  v['year']=year;v['split']=split
  v['user_id']=v.TRAVELER_ID.map(lambda x:year+':'+x if pd.notna(x) else pd.NA)
  v['rating']=pd.to_numeric(v.DGSTFN,errors='coerce')
  norm=lambda x:re.sub(r'\s+','',str(x)).lower() if pd.notna(x) else ''
  v['place_key']=[('poi:'+p) if pd.notna(p) else ('addr:'+norm(a)+'|'+norm(nm)) if pd.notna(a) and pd.notna(nm) else pd.NA for p,a,nm in zip(v.POI_ID,v.ROAD_NM_ADDR,v.VISIT_AREA_NM)]
  v['has_profile']=v.TRAVELER_ID.isin(users.TRAVELER_ID)
  metrics[year+'_'+split]={'visits':len(v),'missing_trip':int(v.TRAVELER_ID.isna().sum()),'missing_profile':int((~v.has_profile).sum()),'duplicate_visit_keys':int(v.duplicated(['TRAVEL_ID','VISIT_AREA_ID']).sum())}
  allv.append(v)
V=pd.concat(allv,ignore_index=True);valid=V.dropna(subset=['user_id','place_key','rating'])
for year in ['2022','2023','combined']:
 d=valid if year=='combined' else valid[valid.year==year]
 p=d.drop_duplicates(['user_id','place_key']);nu=p.user_id.nunique();ni=p.place_key.nunique()
 metrics[year]={'rows':len(d),'pairs':len(p),'users':nu,'places':ni,'sparsity_pct':100*(1-len(p)/(nu*ni)) if nu*ni else None}
V.to_csv(OUT/'joined_visits.csv',index=False)
valid.groupby(['year','split','user_id','place_key'],as_index=False).agg(rating=('rating','mean'),visits=('rating','size')).to_csv(OUT/'interactions.csv',index=False)
(OUT/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2))
print(json.dumps(metrics,ensure_ascii=False,indent=2))
print('Output:',OUT)

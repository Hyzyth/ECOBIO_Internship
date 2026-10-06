"""Validate supplied research frames without committing images or using labels as predictions.

Example: python tools/validate_samples.py --samples /path/Extract --output /tmp/report
Optional --manifest supplies manually reviewed counts and fixture exclusion zones.
"""
import argparse
import json
from pathlib import Path
import sys
import cv2
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backend.services.detection import detect_wells
from backend.services.alignment import align_wells
from backend.services.geometry import is_simple_polygon


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--manifest',type=Path)
    args=parser.parse_args()
    cv2.setNumThreads(2)
    frames=sorted((args.samples/'sequence').glob('*.jpg'))
    if not frames: raise ValueError('No sequence JPEG frames found.')
    manifest=json.loads(args.manifest.read_text()) if args.manifest else {}
    args.output.mkdir(parents=True,exist_ok=True)
    results=[];base_wells=None;reference=None;base_centers=None
    for frame in frames:
        content=frame.read_bytes();image=cv2.imdecode(np.frombuffer(content,np.uint8),cv2.IMREAD_COLOR)
        result=detect_wells(content);wells=result['wells']
        if not wells or not all(is_simple_polygon(w['points']) for w in wells): raise ValueError(f'{frame.name}: no valid wells.')
        expected=manifest.get(frame.name,{})
        if 'complete_well_count' in expected and sum(not w.get('partial',False) for w in wells)!=expected['complete_well_count']: raise ValueError(f'{frame.name}: expected {expected["complete_well_count"]}, detected {len(wells)}.')
        if 'partial_well_count' in expected and sum(w.get('partial',False) for w in wells)!=expected['partial_well_count']:raise ValueError(f'{frame.name}: partial well count differs.')
        centers=np.array([np.mean(w['points'],axis=0) for w in wells])
        for zone in expected.get('fixture_exclusion_zones',[]):
            if np.any(np.linalg.norm(centers-zone['center'],axis=1)<zone['radius']): raise ValueError(f'{frame.name}: mounting fixture proposed as a well.')
        row={'filename':frame.name,'complete_proposals':sum(not w.get('partial',False) for w in wells),'partial_proposals':sum(w.get('partial',False) for w in wells),'method':result['method'],'fixture_zones_checked':len(expected.get('fixture_exclusion_zones',[]))}
        if base_wells is None:
            base_wells=[{'id':str(i),'points':w['points']} for i,w in enumerate(wells)];reference=content;base_centers=centers
        else:
            alignment=align_wells(reference,content,json.dumps(base_wells))
            if [w['id'] for w in alignment['wells']]!=[w['id'] for w in base_wells]:raise ValueError('Alignment changed well identities.')
            aligned=np.array([np.mean(w['points'],axis=0) for w in alignment['wells']])
            if len(centers)!=len(aligned):raise ValueError(f'{frame.name}: independent detection count differs from reference; review manually.')
            errors=np.linalg.norm((aligned-centers)*[1200,1200*image.shape[0]/image.shape[1]],axis=1)
            if np.max(errors)>12:raise ValueError(f'{frame.name}: alignment disagrees with independent detections by more than 12 resized pixels.')
            row['alignment']={k:v for k,v in alignment.items() if k not in ('wells','warning')}
            row['independent_center_agreement_px']={'median':float(np.median(errors)),'max':float(np.max(errors))}
        small=cv2.resize(image,(1200,round(1200*image.shape[0]/image.shape[1])))
        for i,well in enumerate(wells):
            pts=np.int32(np.array(well['points'])*[small.shape[1],small.shape[0]])
            cv2.polylines(small,[pts],True,(255,220,40),2)
            cv2.putText(small,str(i+1),tuple(np.int32(np.mean(pts,axis=0))),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,220,40),1)
        cv2.putText(small,f'{frame.name}: {len(wells)} boundary proposals (including clipped rims); no state predictions',(10,25),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,220,40),1)
        cv2.imwrite(str(args.output/f'{frame.stem}-overlay.jpg'),small)
        results.append(row);print(json.dumps(row),flush=True)
    crops=[]
    for folder in ('coma','awake'):
        for crop in sorted((args.samples/folder).glob('*.png')):
            image=cv2.imread(str(crop))
            if image is None:raise ValueError(f'Unreadable crop: {crop}')
            wells=detect_wells(crop.read_bytes(),layout='individual')['wells']
            if len(wells)!=1 or not is_simple_polygon(wells[0]['points']) or len(wells[0]['points'])<20:
                raise ValueError(f'{crop}: expected one valid rim, not a whole-image fallback.')
            points=np.int32(np.array(wells[0]['points'])*[image.shape[1],image.shape[0]])
            cv2.polylines(image,[points],True,(255,220,40),2)
            cv2.imwrite(str(args.output/f'{folder}-{crop.stem}-overlay.png'),image)
            crops.append({'filename':f'{folder}/{crop.name}','width':image.shape[1],'height':image.shape[0],'boundary':wells[0],'provided_label':folder,'model_prediction':None})
    report={'sequence_frames':results,'crop_inputs':crops,'limitations':['Counts and fixture exclusions require a manually reviewed manifest.','Alignment agreement is relative to independent boundary proposals, not manually annotated ground truth.','Partial rims are fitted from visible evidence and clipped; unseen portions cannot be verified.','Provided crop labels are not predictions and are not used to train a model.']}
    (args.output/'validation.json').write_text(json.dumps(report,indent=2))
    print(f'PASS: {len(results)} frames, {len(crops)} crop inputs. Reports: {args.output}')

if __name__=='__main__':main()

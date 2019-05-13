-- Example code from Eliana meeting on how to burn out delineations for
-- use in organ identification
-- remember to burn out from the deformed image (post-galileo) to remain
-- in the same voxel space

--oars = {'GTV', 'Mandible', 'SC'};
--wm.Scan[4] = wm.Scan[1];
--wm.Scan[4].Data:assign(0)
--for k,v in ipairs(oars) do
  --wm.Scan[3] = wm.Scan[1]:burn(wm.Delineation[v])
  --wm.Scan[4] = wm.Scan[4] + wm.Scan[3]/100*k;
--end
--wm.Scan[4]:write_nifty([[c:\temp\TheOARMap.nii]])

-- first need to perform crop to body delineation to remove excess voxels
-- then burn the resulting body delineation from the 
-- cropping performed in reg program
for x=25,30 do
  wm.Scan[x]:clear()
end

wm.Scan[26] = wm.Scan[20];
wm.Scan[26].Data:assign(0);
wm.Scan[25] = wm.Scan[20]:burn(wm.Delineation.Body,100,false);
wm.Scan[26] = (wm.Scan[25]/100);
wm.Scan[26]:write_nifty([[D:\data\Pancreas\MPhys\panc02BodyBurn.nii]]);
--[[ code to recover one External delineation
wm.Scan[11]:clear()
wm.Scan[12]:clear()
AVS:EULERXFM(wm.Scan[11].Transform)
wm.Scan[11].Data = wm.Scan[1].InverseWarp
wm.Scan[12] = wm.Scan[11];
wm.Scan[12].Data:assign(0);
wm.Scan[12] = (wm.Scan[11]:burn(wm.Delineation.EXTERNAL,100,false))/100;
wm.Scan[12].Transform = wm.Scan[1].Transform
wm.Scan[12]:write_nifty([[C:\MPhys\Nifti_Images\lung104fixed\DVFexternal.nii]]);
]]
wm.Scan[28] = wm.Scan[20];
wm.Scan[28].Data:assign(0);
wm.Scan[27] = wm.Scan[20]:burn(wm.Delineation.BodyMinus,100,false);
wm.Scan[28] = (wm.Scan[27]/100);
wm.Scan[28]:write_nifty([[D:\data\Pancreas\MPhys\panc02Body-0.5Burn.nii]]);

oars = {'Liver','Stomach','SmallBowel','Kidney_L','Kidney_R'};
wm.Scan[30] = wm.Scan[20];
wm.Scan[30].Data:assign(0);
for k,v in ipairs(oars) do
  wm.Scan[29] = wm.Scan[20]:burn(wm.Delineation[v],100,false);
  AVS:FIELD_NORM(wm.Scan[30].Data, wm.Scan[30].Data)
  wm.Scan[30] = wm.Scan[30] + (wm.Scan[29]/100)*k; -- very important to add the scans each time
end
wm.Scan[30]:write_nifty([[D:\data\Pancreas\MPhys\panc02OARMap.nii]]);

--[[
December issue with burning multiple organs:
Burn value defaults to 100, however there seems to be some run off
from each delineation, no matter whether the outer contour is turned
on or not. This 'run off' means that often on the outer regions of the
'higher valued' organs, regions can be misidentified as lower value
organs as the burn value falls into a lower bin in the /100 *k operation
Currently no quick workaround and Eleanor told me to work on the report
instead, guess i should really. Thoughts for the future, singular
organ id works absolutely fine, but multiple organ iding requires 
the run off to be turned off, consult for more.
]]

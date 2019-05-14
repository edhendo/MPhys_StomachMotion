-- 4DCT_stomach_cropping_FINAL_REG.lua
-- MOST RECENT REGISTRATION SCRIPT
-- 3-12 original Scans
-- 13-22 Inverse warps of original scans
-- 23 Average vector field
-- 24 Norm view of average vector field

-- panc01 has maxExhale = 9
-- Stomach02 ME = 10
-- Stomach04 ME = 9
-- Stomach05 ME = 9
-- Stomach06 ME = 9
-- Stomach07 ME = 9
maxExhale = 9
-- set path here
path = [[D:\data\Pancreas\MPhys\Nifti_Images\Stomach_Interpolated\Stomach07\]]
-- first setup and crop around the Stomach_PRV delineation
cropbox = Clipbox:new()
cropbox:fit(wm.Delineation.Stomach,1,7) -- factor=1, dilate=7 to provide boundary around delineation
wm.Scan[maxExhale] = wm.Scan[maxExhale]:crop(cropbox)

averagewarp = field:new()
for j=3, 12 do
  if j ~= maxExhale then
    if not wm.Scan[j].Data.empty then -- empty check
      --wm.Scan[j]:interpolateto(0.35);
      if wm.Scan[j].InverseWarp.empty then wm.scan[j]:niftyreg_f3d(wm.scan[maxExhale],nil,"--rbn 128 --fbn 128 -gpu") end
    else -- Scan[j].Data is empty
      wm.Scan[j] = wm.Scan[j-1] -- copy data from previous Scan
    end
    AVS:FIELD_SLICE(wm.Scan[j].InverseWarp,wm.Scan[j].InverseWarp,-1,-1,-1,0)
    AVS:WARPFIELD_COORDS_DISPL(wm.scan[j].InverseWarp, wm.scan[j].InverseWarp) -- This line is essential for getting vectors not coords
    averagewarp:add(wm.scan[j].InverseWarp) -- sandwich output in here
    AVS:WARPFIELD_DISPL_COORDS(wm.scan[j].InverseWarp, wm.scan[j].InverseWarp) -- VERY IMPORTANT - This line is essential for getting vectors not coords
    print("completed deforming scan", tostring(j))
  end
end
averagewarp = (averagewarp/(-9)) -- average vector field for one patient, i.e average over the number of warps
-- Now extract each individual warp
cropbox2 = Clipbox:new() -- recrop closer to stomach now for data processing
cropbox2:fit(wm.Delineation.Stomach,1,2)
for x=13, 22 do
  if x~=(maxExhale+10) then
    AVS:EULERXFM(wm.Scan[x].Transform)
    AVS:WARPFIELD_COORDS_DISPL(wm.scan[x-10].InverseWarp, wm.scan[x-10].InverseWarp) -- VERY IMPORTANT - This line is essential for getting vectors not coords
    wm.Scan[x].Data = wm.Scan[x-10].InverseWarp
    AVS:WARPFIELD_DISPL_COORDS(wm.scan[x-10].InverseWarp, wm.scan[x-10].InverseWarp) -- VERY IMPORTANT - This line is essential for getting vectors not coords
    AVS:FIELD_ADD(wm.Scan[x].Data,averagewarp,wm.Scan[x].Data)
    wm.Scan[x] = wm.Scan[x]:crop(cropbox2)
    wm.Scan[x]:write_nifty(path .. "warp" .. tostring(x-10) .. ".nii" )
  else
    AVS:EULERXFM(wm.Scan[x].Transform)
    wm.Scan[x].Data = averagewarp
    wm.Scan[x] = wm.Scan[x]:crop(cropbox2)
    wm.Scan[x]:write_nifty(path .. "warp" .. tostring(x-10) .. ".nii")
  end
end

AVS:EULERXFM(wm.Scan[23].Transform)
wm.Scan[23].Data=averagewarp
wm.Scan[23] = wm.Scan[23]:crop(cropbox2)
AVS:EULERXFM(wm.Scan[24].Transform)
AVS:FIELD_NORM(wm.Scan[23].Data, wm.Scan[24].Data)
wm.Scan[24] = wm.Scan[24]:crop(cropbox2)
wm.Scan[23]:write_nifty(path .. "averageVecs.nii")
wm.Scan[24]:write_nifty(path .. "averageNorms.nii")

wm.Scan[1] = wm.Scan[1]:crop(cropbox2)
wm.Scan[2] = wm.Scan[2]:crop(cropbox2)

-- Mask out required delineation (Stomach) 
wm.Scan[2] = wm.Scan[1]:burn(wm.Delineation.Stomach, 1) -- inside 100, outside 0
AVS:FIELD_OPS(wm.Scan[2].Data, wm.Scan[2].Data, 7, AVS.FIELD_OPS_Smooth )
wm.Scan[2].Data:interpolateto(0.35)
wm.Scan[2]:write_nifty(path .. "stomachMask.nii");



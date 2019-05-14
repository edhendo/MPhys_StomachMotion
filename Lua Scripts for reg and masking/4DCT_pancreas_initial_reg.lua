-- 4DCT_pancreas_initial_reg.lua
-- OLD REGISTRATION SCRIPT
-- requires a worldmatch 24
-- 3-12 original Scans
-- 13-22 Inverse warps of original scans
-- 23 Average vector field
-- 24 Norm view of average vector field

-- panc01 has a maximum exhale at frame 9
-- panc02 has a maximum exhale at frame 10
maxExhale = 10

-- first setup and crop around the EXTERNAL delineation
--[[
cropbox = Clipbox:new()
cropbox:fit(wm.Delineation.Body)
for i = 1,12 do
  wm.Scan[i] = wm.Scan[i]:crop(cropbox)
end
]]
averagewarp = field:new() -- average warp is the desired output here
for j=3, 12 do
  if j ~= maxExhale then
    if not wm.Scan[j].Data.empty then -- empty check
      if wm.Scan[j].InverseWarp.empty then wm.scan[j]:galileo(wm.scan[maxExhale]) end
      -- ^^ important reg. line, carries out the galileo function if the inverse warp field is empty
    else -- Scan[j].Data is empty
      wm.Scan[j] = wm.Scan[j-1] -- copy data from previous Scan, (edge case for j=2??)
    end
    AVS:FIELD_SLICE(wm.Scan[j].InverseWarp,wm.Scan[j].InverseWarp,-1,-1,-1,0) -- not sure what this function is
    AVS:WARPFIELD_COORDS_DISPL(wm.scan[j].InverseWarp, wm.scan[j].InverseWarp)
    averagewarp:add(wm.scan[j].InverseWarp) -- add the inverse warp field to the averagewarp field 
    AVS:WARPFIELD_DISPL_COORDS(wm.scan[j].InverseWarp, wm.scan[j].InverseWarp) -- how this differs from above?
    print("completed deforming scan", tostring(j))
  end
end
averagewarp = averagewarp/9 -- average vector field for one patient, i.e average over the number of warps

for x=13, 22 do
  if x~=(maxExhale+10) then
    AVS:EULERXFM(wm.Scan[x].Transform)
    wm.Scan[x].Data = wm.Scan[x-10].InverseWarp
    --wm.Scan[x].Data = averagewarp*-1
    --wm.Scan[x].Data:add(wm.Scan[x-10].InverseWarp)
    wm.Scan[x]:write_nifty([[D:\data\Pancreas\MPhys\Nifti_Images\panc02noAvSubtract\]] .. "warp" .. tostring(x-10) .. ".nii" )
    wm.Scan[x].Data:add(averagewarp*-1)
    wm.Scan[x]:write_nifty([[D:\data\Pancreas\MPhys\Nifti_Images\panc02AvSubtract\]] .. "warp" .. tostring(x-10) .. ".nii" )
  else
    AVS:EULERXFM(wm.Scan[x].Transform)
    wm.Scan[x].Data = averagewarp*-1
    wm.Scan[x]:write_nifty([[D:\data\Pancreas\MPhys\Nifti_Images\panc02noAvSubtract\]] .. "warp" .. tostring(x-10) .. ".nii")
    wm.Scan[x]:write_nifty([[D:\data\Pancreas\MPhys\Nifti_Images\panc02AvSubtract\]] .. "warp" .. tostring(x-10) .. ".nii")
  end
end

AVS:EULERXFM(wm.Scan[23].Transform)
wm.Scan[23].Data=averagewarp
AVS:EULERXFM(wm.Scan[24].Transform)
AVS:FIELD_NORM(wm.Scan[23].Data, wm.Scan[24].Data)

wm.Scan[23]:write_nifty([[D:\data\Pancreas\MPhys\Nifti_Images\panc02noAvSubtract\]] .. "averageVecs.nii")
wm.Scan[24]:write_nifty([[D:\data\Pancreas\MPhys\Nifti_Images\panc02noAvSubtract\]] .. "averageNorms.nii")
wm.Scan[23]:write_nifty([[D:\data\Pancreas\MPhys\Nifti_Images\panc02AvSubtract\]] .. "averageVecs.nii")
wm.Scan[24]:write_nifty([[D:\data\Pancreas\MPhys\Nifti_Images\panc02AvSubtract\]] .. "averageNorms.nii")
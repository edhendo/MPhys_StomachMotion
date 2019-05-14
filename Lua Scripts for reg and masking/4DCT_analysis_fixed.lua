-- 4DCT_pancreas_fixed.lua
-- 05/02/19

-- setup for lung104 with a maximum exhale at frame 7
maxExhale = 7
-- first setup and crop around the EXTERNAL delineation
--[[
cropbox = Clipbox:new()
cropbox:fit(wm.Delineation.EXTERNAL)
for i = 1,10 do
  wm.Scan[i] = wm.Scan[i]:crop(cropbox)
end
]]
averagewarp = field:new() -- average warp is the desired output here
for j=1, 10 do
  if j ~= maxExhale then
    if not wm.Scan[j].Data.empty then -- empty check
      if wm.Scan[j].InverseWarp.empty then wm.scan[j]:galileo(wm.scan[maxExhale]) end
      -- ^^ important reg. line, carries out the galileo function if the inverse warp field is empty
    else -- Scan[j].Data is empty
      wm.Scan[j] = wm.Scan[j-1] -- copy data from previous Scan, (edge case for j=2??)
    end
    AVS:FIELD_SLICE(wm.Scan[j].InverseWarp,wm.Scan[j].InverseWarp,-1,-1,-1,0)
    AVS:WARPFIELD_COORDS_DISPL(wm.scan[j].InverseWarp, wm.scan[j].InverseWarp)
    averagewarp:add(wm.scan[j].InverseWarp) -- sandwich output in here
    AVS:WARPFIELD_DISPL_COORDS(wm.scan[j].InverseWarp, wm.scan[j].InverseWarp) -- VERY IMPORTANT
    print("completed deforming scan", tostring(j))
  end
end
averagewarp = (averagewarp/(-9)) -- average vector field for one patient, i.e average over the number of warps

for x=11, 20 do
  if x~=(maxExhale+10) then
    AVS:EULERXFM(wm.Scan[x].Transform)
    AVS:WARPFIELD_COORDS_DISPL(wm.scan[x-10].InverseWarp, wm.scan[x-10].InverseWarp)
    wm.Scan[x].Data = wm.Scan[x-10].InverseWarp
    AVS:WARPFIELD_DISPL_COORDS(wm.scan[x-10].InverseWarp, wm.scan[x-10].InverseWarp)
    AVS:FIELD_ADD(wm.Scan[x].Data,averagewarp,wm.Scan[x].Data)
    wm.Scan[x]:write_nifty([[C:\MPhys\Nifti_Images\lung104fixed2\]] .. "warp" .. tostring(x-10) .. ".nii" )
  else
    AVS:EULERXFM(wm.Scan[x].Transform)
    wm.Scan[x].Data = averagewarp
    wm.Scan[x]:write_nifty([[C:\MPhys\Nifti_Images\lung104fixed2\]] .. "warp" .. tostring(x-10) .. ".nii")
  end
end

AVS:EULERXFM(wm.Scan[21].Transform)
wm.Scan[21].Data=averagewarp
AVS:EULERXFM(wm.Scan[22].Transform)
AVS:FIELD_NORM(wm.Scan[21].Data, wm.Scan[22].Data)

wm.Scan[21]:write_nifty([[C:\MPhys\Nifti_Images\lung104fixed2\]] .. "averageVecs.nii")
wm.Scan[22]:write_nifty([[C:\MPhys\Nifti_Images\lung104fixed2\]] .. "averageNorms.nii")
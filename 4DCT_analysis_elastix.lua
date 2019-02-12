-- 4DCT_analysis_elastix.lua
  
-- panc01 has maxExhale = 9
maxExhale = 9
-- first setup and crop around the EXTERNAL delineation
--[[
cropbox = Clipbox:new()
cropbox:fit(wm.Delineation.Body)
for i = 3,12 do
  wm.Scan[i] = wm.Scan[i]:crop(cropbox)
end
]]
averagewarp = field:new() -- average warp is the desired output here
for j=3, 12 do
  if j ~= maxExhale then
    if not wm.Scan[j].Data.empty then -- empty check
      if wm.Scan[j].InverseWarp.empty then wm.scan[j]:elastix(wm.scan[maxExhale],1,"par0011\\Parameters.Par0011.bspline1_s.txt") end
    else -- Scan[j].Data is empty
      wm.Scan[j] = wm.Scan[j-1] -- copy data from previous Scan
    end
    AVS:FIELD_SLICE(wm.Scan[j].InverseWarp,wm.Scan[j].InverseWarp,-1,-1,-1,0)
    AVS:WARPFIELD_COORDS_DISPL(wm.scan[j].InverseWarp, wm.scan[j].InverseWarp)
    averagewarp:add(wm.scan[j].InverseWarp) -- sandwich output in here
    AVS:WARPFIELD_DISPL_COORDS(wm.scan[j].InverseWarp, wm.scan[j].InverseWarp) -- VERY IMPORTANT
    print("completed deforming scan", tostring(j))
  end
end
averagewarp = (averagewarp/(-9)) -- average vector field for one patient, i.e average over the number of warps
-- Now extract each individual warp
for x=13, 22 do
  if x~=(maxExhale+10) then
    AVS:EULERXFM(wm.Scan[x].Transform)
    AVS:WARPFIELD_COORDS_DISPL(wm.scan[x-10].InverseWarp, wm.scan[x-10].InverseWarp)
    wm.Scan[x].Data = wm.Scan[x-10].InverseWarp
    AVS:WARPFIELD_DISPL_COORDS(wm.scan[x-10].InverseWarp, wm.scan[x-10].InverseWarp)
    AVS:FIELD_ADD(wm.Scan[x].Data,averagewarp,wm.Scan[x].Data)
    wm.Scan[x]:write_nifty([[D:\data\Pancreas\MPhys\Nifti_Images\panc01fixed\]] .. "warp" .. tostring(x-10) .. ".nii" )
  else
    AVS:EULERXFM(wm.Scan[x].Transform)
    wm.Scan[x].Data = averagewarp
    wm.Scan[x]:write_nifty([[D:\data\Pancreas\MPhys\Nifti_Images\panc01fixed\]] .. "warp" .. tostring(x-10) .. ".nii")
  end
end

AVS:EULERXFM(wm.Scan[23].Transform)
wm.Scan[23].Data=averagewarp
AVS:EULERXFM(wm.Scan[24].Transform)
AVS:FIELD_NORM(wm.Scan[23].Data, wm.Scan[24].Data)

wm.Scan[23]:write_nifty([[D:\data\Pancreas\MPhys\Nifti_Images\panc01fixed\]] .. "averageVecs.nii")
wm.Scan[24]:write_nifty([[D:\data\Pancreas\MPhys\Nifti_Images\panc01fixed\]] .. "averageNorms.nii")
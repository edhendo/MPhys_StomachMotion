-- 4DCT_analysis_elastix.lua
  
-- setup for lung104 with a maximum exhale at frame 7
maxExhale = 7
-- first setup and crop around the EXTERNAL delineation
--cropbox = Clipbox:new()
--cropbox:fit(wm.Delineation.EXTERNAL)
--for i = 3,12 do
  --wm.Scan[i] = wm.Scan[i]:crop(cropbox)
--end

averagewarp = field:new() -- average warp is the desired output here
for j=1, 10 do
  if j ~= maxExhale then
    if not wm.Scan[j].Data.empty then -- empty check
      if wm.Scan[j].InverseWarp.empty then wm.scan[j]:elastix(wm.scan[maxExhale],1,"par0011\\Parameters.Par0011.bspline1_s.txt") end
    end
    print("completed deforming scan", tostring(j))
  end
end
averagewarp = averagewarp/10 -- average vector field for one patient, i.e average over the number of warps

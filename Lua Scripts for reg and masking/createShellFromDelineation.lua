wm.Scan[2] = wm.Scan[1]:burn(wm.Delineation.Stomach, 100 ) -- inside 100, outside 0
AVS:FIELD_OPS(wm.Scan[2].Data, wm.Scan[2].Data, 5, AVS.FIELD_OPS_Smooth )

wm.Scan[2].Data:interpolateto(1)

-- write out the delineation to nifty files here for isosurfacing in python
wm.Scan[2]:write_nifty([[D:\data\Pancreas\MPhys\Nifti_Images\panc01delineations\stomach.nii]])


local dots = Field:new();
local inds = Field:new();
local path = [[D:\data\Pancreas\MPhys\stomachTest.wrl]]
AVS:FIELD_ISOSURFACE( wm.Scan[2].Data, dots, inds, 40 );
-- multipkly dots and inds by transform and adjust at end
AVS:VRML_WRITE( dots, inds, nil, path)

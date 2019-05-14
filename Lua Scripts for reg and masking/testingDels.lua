-- Mask out required delineation (Stomach) 
wm.Scan[1]:interpolateto(0.35)
wm.Scan[2] = wm.Scan[1]:burn(wm.Delineation.Stomach, 100) -- inside 100, outside 0
AVS:FIELD_OPS(wm.Scan[2].Data, wm.Scan[2].Data, 5, AVS.FIELD_OPS_Smooth )

local dots = Field:new();
local inds = Field:new();
local path = [[c:\temp\stomach_0.35.wrl]]
AVS:FIELD_ISOSURFACE( wm.Scan[2].Data, dots, inds, 40 );
-- multipkly dots and inds by transform and adjust at end
AVS:VRML_WRITE( dots, inds, nil, path)


wm.Scan[2].Data:interpolateto(1)
local path = [[c:\temp\stomach_1.00.wrl]]
AVS:FIELD_ISOSURFACE( wm.Scan[2].Data, dots, inds, 40 );
-- multipkly dots and inds by transform and adjust at end
AVS:VRML_WRITE( dots, inds, nil, path)

import sys
from pathlib import Path
import cv2


p = Path(sys.argv[1]).resolve()
img_path = p.as_posix()
export_path = p.parent.joinpath(f'{p.stem}.png').as_posix() 
img = cv2.imread(img_path)
cv2.imwrite(export_path, img)
print(f'Import image: {img_path}')
print(f'Export image: {export_path}')

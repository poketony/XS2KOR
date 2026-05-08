@echo off
@chcp 65001
python ArchivePatchTool.py
XenoLbar.exe xenosaga.00 LBA0_new.txt xenosaga.00.new
python SpliterForxenosaga0.py
ren "xenosaga0.big.new.part1" "xenosaga.01.new"
ren "xenosaga0.big.new.part2" "xenosaga.02.new"
del /q "xenosaga0.big.new"
echo xenosaga0 Repacking Complete
move "xenosaga.00.new" "%USERPROFILE%\Desktop\xenosaga.00.new"
move "xenosaga.01.new" "%USERPROFILE%\Desktop\xenosaga.01.new"
move "xenosaga.02.new" "%USERPROFILE%\Desktop\xenosaga.02.new"
pause
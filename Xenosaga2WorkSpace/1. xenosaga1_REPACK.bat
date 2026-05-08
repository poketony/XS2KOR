@echo off
@chcp 65001
python ArchivePatchTool_1.py
XenoLbar.exe xenosaga.10 LBA1_new.txt xenosaga.10.new
python SpliterForxenosaga1.py
ren "xenosaga1.big.new.part1" "xenosaga.11.new"
ren "xenosaga1.big.new.part2" "xenosaga.12.new"
ren "xenosaga1.big.new.part3" "xenosaga.13.new"
ren "xenosaga1.big.new.part4" "xenosaga.14.new"
del /q "xenosaga1.big.new"
echo xenosaga1 Repacking Complete
move "xenosaga.10.new" "%USERPROFILE%\Desktop\xenosaga.10.new"
move "xenosaga.11.new" "%USERPROFILE%\Desktop\xenosaga.11.new"
move "xenosaga.12.new" "%USERPROFILE%\Desktop\xenosaga.12.new"
move "xenosaga.13.new" "%USERPROFILE%\Desktop\xenosaga.13.new"
move "xenosaga.14.new" "%USERPROFILE%\Desktop\xenosaga.14.new"
pause
set numframes [molinfo top get numframes]
set numatoms [molinfo top get numatoms]
for {set i 0} {$i<$numframes} {incr i} {
  animate goto $i
  puts "Setting User data for frame $i ..."
  set sel [atomselect top all frame $i]
  $sel set user [$sel get occupancy]
}

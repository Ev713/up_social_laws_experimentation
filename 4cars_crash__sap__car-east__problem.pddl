(define (problem sap_intersection-problem)
 (:domain sap_intersection-domain)
 (:objects
   north_ent north_ex east_ent cross_nw south_ex east_ex cross_sw west_ex cross_se west_ent cross_ne south_ent - loc
   north south west east - direction
 )
 (:init (connected south_ent cross_se north) (connected cross_se cross_ne north) (connected cross_ne north_ex north) (connected north_ent cross_nw south) (connected cross_nw cross_sw south) (connected cross_sw south_ex south) (connected east_ent cross_ne west) (connected cross_ne cross_nw west) (connected cross_nw west_ex west) (connected west_ent cross_sw east) (connected cross_sw cross_se east) (connected cross_se east_ex east) (start_ west_ent) (traveldirection east) (free north_ent) (free north_ex) (free east_ent) (free cross_nw) (free south_ex) (free east_ex) (free cross_sw) (free west_ex) (free cross_se) (free west_ent) (free cross_ne) (free south_ent) (not_arrived))
 (:goal (and (at_ east_ex)))
)

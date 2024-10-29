(define (problem sap_intersection-problem)
 (:domain sap_intersection-domain)
 (:objects
   cross_sw south_ex east_ent south_ent north_ex east_ex north_ent cross_se cross_nw cross_ne west_ex west_ent - loc
   north south west east - direction
 )
 (:init (connected south_ent cross_se north) (connected cross_se cross_ne north) (connected cross_ne north_ex north) (connected north_ent cross_nw south) (connected cross_nw cross_sw south) (connected cross_sw south_ex south) (connected east_ent cross_ne west) (connected cross_ne cross_nw west) (connected cross_nw west_ex west) (connected west_ent cross_sw east) (connected cross_sw cross_se east) (connected cross_se east_ex east) (start_ east_ent) (traveldirection west) (free cross_sw) (free south_ex) (free east_ent) (free south_ent) (free north_ex) (free east_ex) (free north_ent) (free cross_se) (free cross_nw) (free cross_ne) (free west_ex) (free west_ent) (not_arrived))
 (:goal (and (at_ west_ex)))
)

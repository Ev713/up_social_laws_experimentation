(define (problem DLOG-5-5-25) (:domain driverlog)
(:objects
	package25 - package
	package24 - package
	package23 - package
	package22 - package
	package21 - package
	package20 - package
	s4 - location
	truck5 - truck
	truck4 - truck
	truck1 - truck
	truck3 - truck
	truck2 - truck
	p17-14 - location
	package18 - package
	s6 - location
	p8-9 - location
	p8-5 - location
	p0-3 - location
	p7-17 - location
	p0-9 - location
	p7-12 - location
	p16-13 - location
	p16-12 - location
	package11 - package
	p4-7 - location
	p5-3 - location
	p13-8 - location
	p4-9 - location
	p3-16 - location
	p17-0 - location
	p14-4 - location
	p14-6 - location
	package16 - package
	p14-0 - location
	package8 - package
	package9 - package
	s18 - location
	package17 - package
	package1 - package
	package2 - package
	package3 - package
	package4 - package
	package5 - package
	package6 - package
	package7 - package
	s9 - location
	s8 - location
	s3 - location
	p19-0 - location
	s1 - location
	s0 - location
	s7 - location
	p6-4 - location
	s5 - location
	p19-1 - location
	package12 - package
	package13 - package
	p18-11 - location
	p19-3 - location
	s11 - location
	p7-3 - location
	s19 - location
	p18-2 - location
	p18-5 - location
	s13 - location
	s12 - location
	p15-18 - location
	s10 - location
	s17 - location
	s16 - location
	s15 - location
	s14 - location
	p10-11 - location
	package19 - package
	p10-15 - location
	package14 - package
	package15 - package
	p13-2 - location
	p14-19 - location
	p12-5 - location
	p9-17 - location
	p11-8 - location
	package10 - package
	p9-13 - location
	s2 - location
	p19-17 - location
	p15-4 - location
	p1-10 - location

	(:private
		driver2 - driver
	)
)
(:init
	(at driver2 s11)
	(at truck1 s4)
	(empty truck1)
	(at truck2 s8)
	(empty truck2)
	(at truck3 s19)
	(empty truck3)
	(at truck4 s0)
	(empty truck4)
	(at truck5 s6)
	(empty truck5)
	(at package1 s19)
	(at package2 s17)
	(at package3 s4)
	(at package4 s10)
	(at package5 s5)
	(at package6 s18)
	(at package7 s7)
	(at package8 s17)
	(at package9 s9)
	(at package10 s2)
	(at package11 s15)
	(at package12 s5)
	(at package13 s8)
	(at package14 s5)
	(at package15 s9)
	(at package16 s19)
	(at package17 s12)
	(at package18 s16)
	(at package19 s11)
	(at package20 s9)
	(at package21 s4)
	(at package22 s18)
	(at package23 s2)
	(at package24 s6)
	(at package25 s1)
	(path s0 p0-3)
	(path p0-3 s0)
	(path s3 p0-3)
	(path p0-3 s3)
	(path s0 p0-9)
	(path p0-9 s0)
	(path s9 p0-9)
	(path p0-9 s9)
	(path s1 p1-10)
	(path p1-10 s1)
	(path s10 p1-10)
	(path p1-10 s10)
	(path s3 p3-16)
	(path p3-16 s3)
	(path s16 p3-16)
	(path p3-16 s16)
	(path s4 p4-7)
	(path p4-7 s4)
	(path s7 p4-7)
	(path p4-7 s7)
	(path s4 p4-9)
	(path p4-9 s4)
	(path s9 p4-9)
	(path p4-9 s9)
	(path s5 p5-3)
	(path p5-3 s5)
	(path s3 p5-3)
	(path p5-3 s3)
	(path s6 p6-4)
	(path p6-4 s6)
	(path s4 p6-4)
	(path p6-4 s4)
	(path s7 p7-3)
	(path p7-3 s7)
	(path s3 p7-3)
	(path p7-3 s3)
	(path s7 p7-12)
	(path p7-12 s7)
	(path s12 p7-12)
	(path p7-12 s12)
	(path s7 p7-17)
	(path p7-17 s7)
	(path s17 p7-17)
	(path p7-17 s17)
	(path s8 p8-5)
	(path p8-5 s8)
	(path s5 p8-5)
	(path p8-5 s5)
	(path s8 p8-9)
	(path p8-9 s8)
	(path s9 p8-9)
	(path p8-9 s9)
	(path s9 p9-13)
	(path p9-13 s9)
	(path s13 p9-13)
	(path p9-13 s13)
	(path s9 p9-17)
	(path p9-17 s9)
	(path s17 p9-17)
	(path p9-17 s17)
	(path s10 p10-11)
	(path p10-11 s10)
	(path s11 p10-11)
	(path p10-11 s11)
	(path s10 p10-15)
	(path p10-15 s10)
	(path s15 p10-15)
	(path p10-15 s15)
	(path s11 p11-8)
	(path p11-8 s11)
	(path s8 p11-8)
	(path p11-8 s8)
	(path s12 p12-5)
	(path p12-5 s12)
	(path s5 p12-5)
	(path p12-5 s5)
	(path s13 p13-2)
	(path p13-2 s13)
	(path s2 p13-2)
	(path p13-2 s2)
	(path s13 p13-8)
	(path p13-8 s13)
	(path s8 p13-8)
	(path p13-8 s8)
	(path s14 p14-0)
	(path p14-0 s14)
	(path s0 p14-0)
	(path p14-0 s0)
	(path s14 p14-4)
	(path p14-4 s14)
	(path s4 p14-4)
	(path p14-4 s4)
	(path s14 p14-6)
	(path p14-6 s14)
	(path s6 p14-6)
	(path p14-6 s6)
	(path s14 p14-19)
	(path p14-19 s14)
	(path s19 p14-19)
	(path p14-19 s19)
	(path s15 p15-4)
	(path p15-4 s15)
	(path s4 p15-4)
	(path p15-4 s4)
	(path s15 p15-18)
	(path p15-18 s15)
	(path s18 p15-18)
	(path p15-18 s18)
	(path s16 p16-12)
	(path p16-12 s16)
	(path s12 p16-12)
	(path p16-12 s12)
	(path s16 p16-13)
	(path p16-13 s16)
	(path s13 p16-13)
	(path p16-13 s13)
	(path s17 p17-0)
	(path p17-0 s17)
	(path s0 p17-0)
	(path p17-0 s0)
	(path s17 p17-14)
	(path p17-14 s17)
	(path s14 p17-14)
	(path p17-14 s14)
	(path s18 p18-2)
	(path p18-2 s18)
	(path s2 p18-2)
	(path p18-2 s2)
	(path s18 p18-5)
	(path p18-5 s18)
	(path s5 p18-5)
	(path p18-5 s5)
	(path s18 p18-11)
	(path p18-11 s18)
	(path s11 p18-11)
	(path p18-11 s11)
	(path s19 p19-0)
	(path p19-0 s19)
	(path s0 p19-0)
	(path p19-0 s0)
	(path s19 p19-1)
	(path p19-1 s19)
	(path s1 p19-1)
	(path p19-1 s1)
	(path s19 p19-3)
	(path p19-3 s19)
	(path s3 p19-3)
	(path p19-3 s3)
	(path s19 p19-17)
	(path p19-17 s19)
	(path s17 p19-17)
	(path p19-17 s17)
	(link s0 s2)
	(link s2 s0)
	(link s0 s13)
	(link s13 s0)
	(link s0 s16)
	(link s16 s0)
	(link s0 s18)
	(link s18 s0)
	(link s1 s9)
	(link s9 s1)
	(link s1 s11)
	(link s11 s1)
	(link s2 s6)
	(link s6 s2)
	(link s2 s10)
	(link s10 s2)
	(link s2 s12)
	(link s12 s2)
	(link s2 s15)
	(link s15 s2)
	(link s3 s13)
	(link s13 s3)
	(link s3 s14)
	(link s14 s3)
	(link s3 s17)
	(link s17 s3)
	(link s3 s19)
	(link s19 s3)
	(link s4 s1)
	(link s1 s4)
	(link s4 s2)
	(link s2 s4)
	(link s4 s7)
	(link s7 s4)
	(link s5 s10)
	(link s10 s5)
	(link s5 s14)
	(link s14 s5)
	(link s5 s17)
	(link s17 s5)
	(link s6 s3)
	(link s3 s6)
	(link s6 s10)
	(link s10 s6)
	(link s6 s11)
	(link s11 s6)
	(link s6 s12)
	(link s12 s6)
	(link s6 s19)
	(link s19 s6)
	(link s7 s3)
	(link s3 s7)
	(link s8 s0)
	(link s0 s8)
	(link s8 s3)
	(link s3 s8)
	(link s8 s13)
	(link s13 s8)
	(link s8 s18)
	(link s18 s8)
	(link s9 s2)
	(link s2 s9)
	(link s9 s10)
	(link s10 s9)
	(link s10 s13)
	(link s13 s10)
	(link s10 s18)
	(link s18 s10)
	(link s11 s3)
	(link s3 s11)
	(link s11 s4)
	(link s4 s11)
	(link s11 s5)
	(link s5 s11)
	(link s11 s8)
	(link s8 s11)
	(link s11 s9)
	(link s9 s11)
	(link s11 s17)
	(link s17 s11)
	(link s11 s18)
	(link s18 s11)
	(link s11 s19)
	(link s19 s11)
	(link s12 s0)
	(link s0 s12)
	(link s12 s1)
	(link s1 s12)
	(link s12 s5)
	(link s5 s12)
	(link s12 s9)
	(link s9 s12)
	(link s12 s10)
	(link s10 s12)
	(link s12 s11)
	(link s11 s12)
	(link s13 s1)
	(link s1 s13)
	(link s13 s11)
	(link s11 s13)
	(link s14 s6)
	(link s6 s14)
	(link s14 s17)
	(link s17 s14)
	(link s15 s5)
	(link s5 s15)
	(link s15 s9)
	(link s9 s15)
	(link s15 s12)
	(link s12 s15)
	(link s16 s2)
	(link s2 s16)
	(link s16 s5)
	(link s5 s16)
	(link s16 s7)
	(link s7 s16)
	(link s16 s10)
	(link s10 s16)
	(link s17 s7)
	(link s7 s17)
	(link s17 s19)
	(link s19 s17)
	(link s18 s1)
	(link s1 s18)
	(link s18 s4)
	(link s4 s18)
	(link s18 s7)
	(link s7 s18)
	(link s18 s14)
	(link s14 s18)
)
(:goal
	(and
		(at truck1 s2)
		(at truck2 s11)
		(at truck3 s10)
		(at truck4 s3)
		(at truck5 s16)
		(at package1 s19)
		(at package2 s10)
		(at package3 s19)
		(at package4 s11)
		(at package5 s14)
		(at package6 s18)
		(at package7 s7)
		(at package8 s6)
		(at package9 s7)
		(at package10 s14)
		(at package11 s13)
		(at package12 s11)
		(at package13 s15)
		(at package14 s6)
		(at package15 s11)
		(at package16 s10)
		(at package17 s17)
		(at package18 s15)
		(at package19 s4)
		(at package20 s7)
		(at package21 s3)
		(at package22 s8)
		(at package23 s17)
		(at package24 s2)
		(at package25 s12)
	)
)
)
---
layout: post
title: DB SQL 알아보기 
date: 2021-08-28 00:00:00
img: etc/db/sql_basic/0.png
categories: [etc-db] 
tags: [Database, SQL] # add tag
---

<br>

- 이번 글에서는 DataBase의 교과서라고 불리는 [Database System Concepts](https://www.db-book.com/db7/)의 예제를 이용하여 기본적인 `SQL`에 대하여 알아보도록 하겠습니다.
- ① SQL 실습 방법 : [https://www.db-book.com/db7/university-lab-dir/sqljs.html](https://www.db-book.com/db7/university-lab-dir/sqljs.html)
- ② MySQL 또는 PostgreSQL에서 실습하기 위한 Table 및 샘플 DB 선언 후 사용 시 글 마지막의 SQL문을 사용하시면 됩니다.

<br>

## **Aggregate Functions**

<br>

- 이번에는 `Aggregate Function`에 대하여 알아보도록 하겠습니다.
- `Aggregate Function`은 Relation의 Column 값에 대해 동작하고 결과 값을 반환합니다. 대표적으로 `avg`, `min`, `max`, `sum`, `count`가 있습니다.
- 먼저 `avg`를 이용하여 어떻게 동작하는 지 살펴보도록 하겠습니다.

<br>

```sql
select salary
from instructor
where dept_name = 'Comp. Sci.'

-- salary
-- 65000
-- 75000
-- 92000

select avg(salary)
from instructor
where dept_name = 'Comp. Sci.'

-- avg(salary)
-- 77333.33333333333
```

<br>

- 위 코드와 같이 aggregate function인 `avg`를 사용하면 결과값을 요약하는 연산(평균)이 적용된 결과가 나오게 됩니다.

<br>

- Spring, 2018 semester에서 강의를 한 instructor의 전체 수를 구하려면 다음과 같이 구할 수 있습니다.

<br>

```sql
select count(distinct ID)
from teaches
where semester = 'Spring' and year = 2018;

-- count(distinct ID)
-- 6
```

<br>


<br>

## **Nested Subqueries**

<br>

- `SQL`에서는 Query 안에서 또다른 `select-from-where`를 가지는 sub query를 가질 수 있습니다.
- 다음과 


<br>

## **Table 및 샘플 DB 선언을 위한 SQL**

<br>

```sql
DROP TABLE IF EXISTS `classroom`;

CREATE TABLE `classroom` (
  `building` varchar(15) NOT NULL,
  `room_number` varchar(7) NOT NULL,
  `capacity` int DEFAULT NULL,
  PRIMARY KEY (`building`,`room_number`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

INSERT INTO `classroom` VALUES ('Packard','101',500),('Painter','514',10),('Taylor','3128',70),('Watson','100',30),('Watson','120',50);

DROP TABLE IF EXISTS `department`;

CREATE TABLE `department` (
	`dept_name` varchar(20) NOT NULL,
	`building` varchar(15) DEFAULT NULL,
	`budget` decimal(12,2) DEFAULT NULL,
	PRIMARY KEY (`dept_name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

INSERT INTO `department` VALUES ('Biology','Watson',90000.00),('Comp. Sci.','Taylor',100000.00),('Elec. Eng.','Taylor',85000.00),('Finance','Painter',120000.00),('History','Painter',50000.00),('Music','Packard',80000.00),('Physics','Watson',70000.00);

DROP TABLE IF EXISTS `course`;

CREATE TABLE `course` (
  `course_id` varchar(8) NOT NULL,
  `title` varchar(50) DEFAULT NULL,
  `dept_name` varchar(20) NOT NULL,
  `credits` int DEFAULT NULL,
  PRIMARY KEY (`course_id`),
  FOREIGN KEY (`dept_name`) references `department`(`dept_name`) on delete RESTRICT
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

INSERT INTO `course` VALUES ('BIO-101','Intro. to Biology','Biology',4),('BIO-301','Genetics','Biology',4),('BIO-399','Computational Biology','Biology',3),('CS-101','Intro. to Computer Science','Comp. Sci.',4),('CS-190','Game Design','Comp. Sci.',4),('CS-315','Robotics','Comp. Sci.',3),('CS-319','Image Processing','Comp. Sci.',3),('CS-347','Database System Concepts','Comp. Sci.',3),('EE-181','Intro. to Digital Systems','Elec. Eng.',3),('FIN-201','Investment Banking','Finance',3),('HIS-351','World History','History',3),('MU-199','Music Video Production','Music',3),('PHY-101','Physical Principles','Physics',4);

DROP TABLE IF EXISTS `instructor`;

CREATE TABLE `instructor` (
	`ID` char(5)  NOT NULL,
	`name` varchar(20) NOT NULL,
	`dept_name`  varchar(20) NOT NULL, 
	`salary` decimal(8,2) DEFAULT NULL, 
	PRIMARY KEY (`ID`),            
	FOREIGN KEY (`dept_name`) references `department`(`dept_name`) on delete RESTRICT
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

INSERT INTO `instructor` VALUES ('10101','Srinivasan','Comp. Sci.',65000.00),('12121','Wu','Finance',90000.00),('15151','Mozart','Music',40000.00),('22222','Einstein','Physics',95000.00),('32343','El Said','History',60000.00),('33456','Gold','Physics',87000.00),('45565','Katz','Comp. Sci.',75000.00),('58583','Califieri','History',62000.00),('76543','Singh','Finance',80000.00),('76766','Crick','Biology',72000.00),('83821','Brandt','Comp. Sci.',92000.00),('98345','Kim','Elec. Eng.',80000.00);

DROP TABLE IF EXISTS `time_slot`;

CREATE TABLE `time_slot` (
  `time_slot_id` varchar(4) NOT NULL,
  `day` varchar(1) NOT NULL,
  `start_hr` decimal(2,0) NOT NULL,
  `start_min` decimal(2,0) NOT NULL,
  `end_hr` decimal(2,0) NOT NULL,
  `end_min` decimal(2,0) NOT NULL,
  PRIMARY KEY (`time_slot_id`,`day`,`start_hr`,`start_min`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

INSERT INTO `time_slot` VALUES ('A','F',8,0,8,50),('A','M',8,0,8,50),('A','W',8,0,8,50),('B','F',9,0,9,50),('B','M',9,0,9,50),('B','W',9,0,9,50),('C','F',11,0,11,50),('C','M',11,0,11,50),('C','W',11,0,11,50),('D','F',13,0,13,50),('D','M',13,0,13,50),('D','W',13,0,13,50),('E','R',10,30,11,45),('E','T',10,30,11,45),('F','R',14,30,15,45),('F','T',14,30,15,45),('G','F',16,0,16,50),('G','M',16,0,16,50),('G','W',16,0,16,50),('H','W',10,0,12,30);

DROP TABLE IF EXISTS `section`;

CREATE TABLE `section` (
  `course_id` varchar(8) NOT NULL,
  `sec_id` varchar(8) NOT NULL,
  `semester` varchar(6) NOT NULL,
  `year` int NOT NULL,
  `building` varchar(15) NOT NULL,
  `room_number` varchar(7) NOT NULL,
  `time_slot_id` varchar(4) NOT NULL,
  PRIMARY KEY (`course_id`,`sec_id`,`semester`,`year`),
  FOREIGN KEY (`course_id`) references `course`(`course_id`) on delete RESTRICT,
  FOREIGN KEY (`building`, `room_number`) references `classroom`(`building`, `room_number`) on delete RESTRICT,
  FOREIGN KEY (`time_slot_id`) references `time_slot`(`time_slot_id`) on delete RESTRICT
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

INSERT INTO `section` VALUES ('BIO-101','1','Summer',2009,'Painter','514','B'),('BIO-301','1','Summer',2010,'Painter','514','A'),('CS-101','1','Fall',2009,'Packard','101','H'),('CS-101','1','Spring',2010,'Packard','101','F'),('CS-190','1','Spring',2009,'Taylor','3128','E'),('CS-190','2','Spring',2009,'Taylor','3128','A'),('CS-315','1','Spring',2010,'Watson','120','D'),('CS-319','1','Spring',2010,'Watson','100','B'),('CS-319','2','Spring',2010,'Taylor','3128','C'),('CS-347','1','Fall',2009,'Taylor','3128','A'),('EE-181','1','Spring',2009,'Taylor','3128','C'),('FIN-201','1','Spring',2010,'Packard','101','B'),('HIS-351','1','Spring',2010,'Painter','514','C'),('MU-199','1','Spring',2010,'Packard','101','D'),('PHY-101','1','Fall',2009,'Watson','100','A');

DROP TABLE IF EXISTS `teaches`;

CREATE TABLE `teaches` (
  `ID` varchar(5) NOT NULL,
  `course_id` varchar(8) NOT NULL,
  `sec_id` varchar(8) NOT NULL,
  `semester` varchar(6) NOT NULL,
  `year` int NOT NULL,
  PRIMARY KEY (`ID`,`course_id`,`sec_id`,`semester`,`year`),
  FOREIGN KEY (`course_id`,`sec_id`,`semester`,`year`) references `section`(`course_id`,`sec_id`,`semester`,`year`) on delete restrict,
  FOREIGN KEY (`ID`) references `instructor`(`ID`) on delete RESTRICT
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

INSERT INTO `teaches` VALUES ('10101','CS-101','1','Fall',2009),('10101','CS-315','1','Spring',2010),('10101','CS-347','1','Fall',2009),('12121','FIN-201','1','Spring',2010),('15151','MU-199','1','Spring',2010),('22222','PHY-101','1','Fall',2009),('32343','HIS-351','1','Spring',2010),('45565','CS-101','1','Spring',2010),('45565','CS-319','1','Spring',2010),('76766','BIO-101','1','Summer',2009),('76766','BIO-301','1','Summer',2010),('83821','CS-190','1','Spring',2009),('83821','CS-190','2','Spring',2009),('83821','CS-319','2','Spring',2010),('98345','EE-181','1','Spring',2009);

DROP TABLE IF EXISTS `student`;

CREATE TABLE `student` (
  `ID` varchar(5) NOT NULL,
  `name` varchar(20) NOT NULL,
  `dept_name` varchar(20),
  `tot_cred` int,
  PRIMARY KEY (`ID`),
  FOREIGN KEY (`dept_name`) references `department`(`dept_name`) on delete RESTRICT
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

INSERT INTO `student` VALUES ('00000', 'ShinHwan Kang', 'Comp. Sci.', 100), ('00001', 'HoeHoon Jung', 'Comp. Sci.', 100), ('00128','Zhang','Comp. Sci.',102),('12345','Shankar','Comp. Sci.',32),('19991','Brandt','History',80),('23121','Chavez','Finance',110),('44553','Peltier','Physics',56),('45678','Levy','Physics',46),('54321','Williams','Comp. Sci.',54),('55739','Sanchez','Music',38),('70557','Snow','Physics',0),('76543','Brown','Comp. Sci.',58),('76653','Aoi','Elec. Eng.',60),('98765','Bourikas','Elec. Eng.',98),('98988','Tanaka','Biology',120);

DROP TABLE IF EXISTS `advisor`;

CREATE TABLE `advisor` (
  `s_ID` varchar(5) NOT NULL,
  `i_ID` varchar(5) NOT NULL,
  PRIMARY KEY (`s_ID`),
  FOREIGN KEY (`i_ID`) REFERENCES `instructor` (`ID`) on DELETE RESTRICT,
  FOREIGN KEY (`s_ID`) REFERENCES `student` (`ID`) on DELETE RESTRICT
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

INSERT INTO `advisor` (`s_ID`, `i_ID`) VALUES
('12345', '10101'),
('44553', '22222'),
('45678', '22222'),
('00128', '45565'),
('76543', '45565'),
('23121', '76543'),
('98988', '76766'),
('76653', '98345'),
('98765', '98345');

DROP TABLE IF EXISTS `takes`;

CREATE TABLE `takes` (
  `ID` varchar(5) NOT NULL,
  `course_id` varchar(8) NOT NULL,
  `sec_id` varchar(8) NOT NULL,
  `semester` varchar(6) NOT NULL,
  `year` int NOT NULL,
  `grade` varchar(2) DEFAULT NULL,
  PRIMARY KEY (`ID`,`course_id`,`sec_id`,`semester`,`year`),
  FOREIGN KEY (`ID`) REFERENCES `student` (`ID`) on DELETE RESTRICT,
  FOREIGN KEY (`course_id`,`sec_id`,`semester`,`year`) references `section`(`course_id`,`sec_id`,`semester`,`year`) on delete restrict
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

INSERT INTO `takes` VALUES ('00000', 'CS-347', '1', 'Fall', '2009', 'A+'), ('00001', 'CS-347', '1', 'Fall', '2009', 'A+'), ('00128','CS-101','1','Fall',2009,'A'),('00128','CS-347','1','Fall',2009,'A-'),('12345','CS-101','1','Fall',2009,'C'),('12345','CS-190','2','Spring',2009,'A'),('12345','CS-315','1','Spring',2010,'A'),('12345','CS-347','1','Fall',2009,'A'),('19991','HIS-351','1','Spring',2010,'B'),('23121','FIN-201','1','Spring',2010,'C+'),('44553','PHY-101','1','Fall',2009,'B-'),('45678','CS-101','1','Fall',2009,'F'),('45678','CS-101','1','Spring',2010,'B+'),('45678','CS-319','1','Spring',2010,'B'),('54321','CS-101','1','Fall',2009,'A-'),('54321','CS-190','2','Spring',2009,'B+'),('55739','MU-199','1','Spring',2010,'A-'),('76543','CS-101','1','Fall',2009,'A'),('76543','CS-319','2','Spring',2010,'A'),('76653','EE-181','1','Spring',2009,'C'),('98765','CS-101','1','Fall',2009,'C-'),('98765','CS-315','1','Spring',2010,'B'),('98988','BIO-101','1','Summer',2009,'A'),('98988','BIO-301','1','Summer',2010,NULL);

DROP TABLE IF EXISTS `prereq`;

CREATE TABLE `prereq` (
  `course_id` varchar(8) NOT NULL,
  `prereq_id` varchar(8) NOT NULL,
  PRIMARY KEY (`course_id`,`prereq_id`),
  FOREIGN KEY (`course_id`) REFERENCES `course` (`course_id`) on DELETE RESTRICT,
  FOREIGN KEY (`prereq_id`) REFERENCES `course` (`course_id`) on DELETE RESTRICT
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

INSERT INTO `prereq` VALUES ('BIO-301','BIO-101'),('BIO-399','BIO-101'),('CS-190','CS-101'),('CS-315','CS-101'),('CS-319','CS-101'),('CS-347','CS-101'),('EE-181','PHY-101');
```
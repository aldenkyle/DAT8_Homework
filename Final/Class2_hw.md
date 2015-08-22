## Class 2 Homework 

1_ Each row represents one *item* ordered as part of a greater order at Chipotle,

2_ 1834 

3_ 4623

4_ Chicken is more popular than steak. (Chicken = 553, Steak = 368)

5_ Black beans are more popular. (Black Beans = 282, Pinto Beans = 105)

6_ I used: find . -name *.?sv to get:
	*./data/airlines.csv*
	*./data/chipotle.tsv*
	*./data/sms.tsv*

7_ The word “dictionary” was found 10 times in the DAT8 repo.

8_ The most common item ordered was the Chicken Bowl. Here is a ranked list and the code that I used: 

kyle-aldens-macbook-2:data kylealden$ cut -f3 chipotle.tsv | sort | uniq -c | sort -r

 726 Chicken Bowl
 553 Chicken Burrito
 479 Chips and Guacamole
 368 Steak Burrito
 301 Canned Soft Drink
 211 Steak Bowl
 211 Chips
 162 Bottled Water
 115 Chicken Soft Tacos
 110 Chips and Fresh Tomato Salsa
 110 Chicken Salad Bowl
 104 Canned Soda
 101 Side of Chips
  95 Veggie Burrito
  91 Barbacoa Burrito
  85 Veggie Bowl
  68 Carnitas Bowl
  66 Barbacoa Bowl
  59 Carnitas Burrito
  55 Steak Soft Tacos
  54 6 Pack Soft Drink
  48 Chips and Tomatillo Red Chili Salsa
  47 Chicken Crispy Tacos
  43 Chips and Tomatillo Green Chili Salsa
  40 Carnitas Soft Tacos
  35 Steak Crispy Tacos
  31 Chips and Tomatillo-Green Chili Salsa
  29 Steak Salad Bowl
  27 Nantucket Nectar
  25 Barbacoa Soft Tacos
  22 Chips and Roasted Chili Corn Salsa
  20 Izze
  20 Chips and Tomatillo-Red Chili Salsa
  18 Veggie Salad Bowl
  18 Chips and Roasted Chili-Corn Salsa
  11 Barbacoa Crispy Tacos
  10 Barbacoa Salad Bowl
   9 Chicken Salad
   7 Veggie Soft Tacos
   7 Carnitas Crispy Tacos
   6 Veggie Salad
   6 Carnitas Salad Bowl
   6 Burrito
   4 Steak Salad
   2 Salad
   2 Crispy Tacos
   2 Bowl
   1 item_name
   1 Veggie Crispy Tacos
   1 Chips and Mild Fresh Tomato Salsa
   1 Carnitas Salad

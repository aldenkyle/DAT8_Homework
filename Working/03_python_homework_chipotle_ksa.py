'''
Python Homework with Chipotle data
https://github.com/TheUpshot/chipotle
'''

'''
BASIC LEVEL
PART 1: Read in the file with csv.reader() and store it in an object called 'file_nested_list'.
Hint: This is a TSV file, and csv.reader() needs to be told how to handle it.
      https://docs.python.org/2/library/csv.html
'''

import csv
with open('chipotle.tsv', mode = 'rU') as f:
    file_nested_list = [row for row in csv.reader(f, delimiter='\t')]



'''
BASIC LEVEL
PART 2: Separate 'file_nested_list' into the 'header' and the 'data'.
'''
header = file_nested_list[0]
data = file_nested_list[1:]

'''
INTERMEDIATE LEVEL
PART 3: Calculate the average price of an order.
Hint: Examine the data to see if the 'quantity' column is relevant to this calculation.
Hint: Think carefully about the simplest way to do this!
'''
### i found the sum of all items by multipying the item_price by 
### the quantity. then divided it by the number of orders
### which i found by grabbing the last row
### The average price was per order was $21.39

orders_sum = 0.0
for order in data:
    amount = order[4].replace('$','')
    orders_sum += float(order[1])*float(amount)

print(orders_sum)    

average = orders_sum / 1834

print(average)

'''
INTERMEDIATE LEVEL
PART 4: Create a list (or set) of all unique sodas and soft drinks that they sell.
Note: Just look for 'Canned Soda' and 'Canned Soft Drink', and ignore other drinks like 'Izze'.
'''
all_sodas = []
for order in data:
    if order[2] == 'Canned Soda':
        all_sodas.append(order[3])
    elif order[2] == 'Canned Soft Drink':
        all_sodas.append(order[3])
        
print(all_sodas)

soda_types = set(all_sodas)

print(soda_types)

## The list of sodas and soft drinks that were sold is:['[Lemonade]', '[Dr. Pepper]',
## '[Diet Coke]', '[Nestea]', '[Mountain Dew]', '[Diet Dr. Pepper]', 
##'[Coke]', '[Coca Cola]', '[Sprite]']

'''
ADVANCED LEVEL
PART 5: Calculate the average number of toppings per burrito.
Note: Let's ignore the 'quantity' column to simplify this task.
Hint: Think carefully about the easiest way to count the number of toppings!
'''

## counted up the number of toppings by counting the commas in each 
##'choice description' row and adding 1 to to it. Counted up the 
## number of burritos using the same for loop if 'Burrito' was in a row
toppings_sum = 0
burrito_count = 0
for row in data:
    if 'Burrito' in row[2]:
        toppings_sum += (row[3].count(',') + 1)
        burrito_count += 1
        
print(toppings_sum)
print(burrito_count)

## found average by dividing the total toppings by the count of burritos. 
average_toppings = float(toppings_sum) / burrito_count

print(average_toppings)

## The average number of toppings per burrito was 5.395 when 
## we ignored the quantity column. 

'''
ADVANCED LEVEL
PART 6: Create a dictionary in which the keys represent chip orders and
  the values represent the total number of orders.
Expected output: {'Chips and Roasted Chili-Corn Salsa': 18, ... }
Note: Please take the 'quantity' column into account!
Optional: Learn how to use 'defaultdict' to simplify your code.
'''

## First set up new list of types of chip orders by looping through the data,
## pulling out everything with "chip" in the item name, and then using set
chips_all = []
for row in data:
    if "Chip" in row[2]:
        chips_all.append(row[2])
chips_set = list(set(chips_all))
print(chips_set)

##Then created a dictionary using nested for loops that loop through
## both the new list of chip orders, and then through the full data, and 
##counts up the sum of the order quantities. 
mdict = {}
set_num = 0
for chip in chips_set:
    orders = 0
    for row in data: 
        if row[2] == chips_set[set_num]:
            orders += int(row[1])
            mdict[chips_set[set_num]] = orders
    set_num += 1

print(mdict)

## Dictionary of chip order types and number of orders (added carraige returns for readability):
##{'Chips and Roasted Chili-Corn Salsa': 18, 'Chips and Mild Fresh Tomato Salsa': 1,
##'Chips and Tomatillo-Red Chili Salsa': 25, 'Chips and Guacamole': 506, 
##'Chips and Fresh Tomato Salsa': 130, 'Side of Chips': 110, 
##'Chips and Tomatillo-Green Chili Salsa': 33, 
##'Chips and Tomatillo Red Chili Salsa': 50, 'Chips and Roasted Chili Corn Salsa': 23, 
##'Chips': 230, 'Chips and Tomatillo Green Chili Salsa': 45}

'''
BONUS: Think of a question about this data that interests you, and then answer it!
'''
##how many orders contained both salads and chips?
## Create list of orders that contain salads:

## created a list of orders the contain salad
salad_orders = []
for row in data:
    if "Salad" in row[2]:
        salad_orders.append(row[0])

print(len(salad_orders))

## created a list of orders that contained chips that were already 
##identified as containing salads

salad_and_chip_eaters = []
for row in data:
    if row[0] in salad_orders:
        if "Chip" in row[2]:
            salad_and_chip_eaters.append(row[0])
            
print(len(salad_and_chip_eaters))


## 71 out of 195 orders that contained salads also contained chips (36%)

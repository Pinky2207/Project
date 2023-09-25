from math import ceil


def non_prime(a,b):
  # Set upper and lower as the min/max of a and b
  lower = min(a, b)
  upper = max(a, b)

  # Create an empty list for non-prime numbers
  non_prime_nums_list = []

  # Loop through each number between lower and upper (inclusive)
  for number in range(lower, upper+1):
    # Check if the number is greater than or equal to 1
    if number >= 1:
      # Loop through each index between 2 and the number (exclusive)
      for index in range(2, number):
        # Check if the number is divisible by the index with no remainder
        if (number % index) == 0:
          # If the number is divisible by the index, 
          # add it to the non_prime_nums_list 
          non_prime_nums_list.append(number)
          break

  # Return the list of non-prime numbers
  return non_prime_nums_list


def user_input():
    # Loop until valid input is entered
    while True:
        # Ask the user to enter the first number
        num_1 = input("Enter the number1 :")
        
        # Ask the user to enter the second number
        num_2 = input("Enter the number2 :")
        
        # Check if both inputs are numeric
        if num_1.isnumeric() and num_2.isnumeric():
            print("Given input value is correct")
            # Convert both inputs to integers
            num_1 = int(num_1)
            num_2 = int(num_2)
        else:
            # If either input is not numeric, print an error message and continue to the next iteration of the loop
            print("Entered value is incorrect")
            break

        # Call the non_prime() function to get a list of non-prime numbers in the range between num_1 and num_2
        list_of_non_primes = non_prime(num_1, num_2)
        
        # Calculate the number of lines needed to print all the non-prime numbers, with each line containing at most 10 numbers
        number_of_lines = ceil(len(list_of_non_primes) / 10)
        start = 0
        end = 10
        
        # Loop through each line of output
        for line in range(number_of_lines):
            # Get the next 10 non-prime numbers to print
            output = list_of_non_primes[start:end]
            
            # Print the non-prime numbers on this line, separated by spaces
            print(*output)
            
            # Update the start and end indexes for the next line
            start = end
            end = end+10

        # Break out of the loop after successfully processing user input and printing output
        break



user_input()

  

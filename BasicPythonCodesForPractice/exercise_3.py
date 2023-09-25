
# This function checks if a given string is a palindrome
def palindrome(a):
  # Check if the input string is equal to its reverse
  if a == a[::-1]:
    # If the string is a palindrome, print True
    return True
  else:
    # If the string is not a palindrome, print False
    return False


def frequent_letter_digit(b):

  # Converting the input string to upper case
  b = b.upper()
  # Printing the converted input
  print(b)

# Creating empty dictionary to store the frequent letters or digits
  frequency = {}

  for character in b:
    # Only considers alphanumeric characters and ignores the rest 
    if character.isalnum(): 

      if character in frequency:
        # if character already in the dictionary then the count is incremented 
        frequency[character] = frequency[character] + 1

      # Else if already not in the dictionary yet then it is returned with a count of 1
      else:
        frequency[character] = 1

  #  Finding the most frequently occured letter or digit
  most_frequent = max(frequency, key=frequency.get)

  return most_frequent


def count_letters_spaces_digits(e):
# Assigning the initial values as 0
  letter = 0
  space = 0 
  digit = 0

  # Iterating through each character in the given input
  for char in e:

    # If the character is letter 
    if char.isalpha():
      letter = letter + 1

    # if the input character contains space 
    elif char.isspace():
      space = space + 1

    # if the input character is digits
    elif char.isdigit():
      digit = digit + 1
      
# Summarizing the total number of letters, spaces and digits
  counts = {'No of letters': letter, 'No of spaces': space, 'No of digits': digit}
  return counts

print(count_letters_spaces_digits('Archit Dubey 10 !@#$%'))
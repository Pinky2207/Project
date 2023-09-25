import datetime
from datetime import date


american_dob_format = '%m/%d/%Y'
euro_dob_format = '%d/%m/%Y'


def date_of_birth_validation(dob_input):
    try:
        # try to validate dob_input
        dob_validation = datetime.datetime.strptime(dob_input, american_dob_format)

    except ValueError:
        # raise error if dob_input is wrong
        raise ValueError("Error: Invalid date format")
        
    else:
        # if input is correct, return dob_input
        return dob_input


def calculate_age(valid_dob_input):
    while True: 
    
        # convert string dob_input to python date object using format
        dob = datetime.datetime.strptime(valid_dob_input, american_dob_format).date()
        # get today's date
        current_date = date.today()
            # if dob is greater than today, raise an error
        if dob > current_date:
            raise ValueError("DOB cannot be future date.")
        break
            
        # get birthday
    birthday = dob.replace(year=current_date.year)

        # subtract dob year from current year as age
    age = current_date.year - dob.year

        # if today is less than birthday, subtract 1 from age
    if current_date < birthday:
            age = age - 1

        # return age
    return age


def start():
    
    while True:
        # ask for input
        dob_input = input("Please enter your date of birth in the format mm/dd/yyyy: ")
        # validate input
        try:
            valid_dob = date_of_birth_validation(dob_input)
        except ValueError as error:
            print(error)
            continue

        # if input is valid, calculate age
        try:
            user_age = calculate_age(valid_dob)
        except ValueError as error:
            print(error)
            continue

        # M/D/Y
        month, day, year = valid_dob.split('/')
        eur_dob = f"{day}/{month}/{year}"

        print(user_age, eur_dob)
        break


start()
from tabulate import tabulate   

# This function returns the salary of an employee, given a tuple containing employee information.
def employee_salary(info):
    return info[2]

# This function filters the employee information for those with salaries within a given range and returns the sorted results.
def employee_details(info, minimum, maximum):
    salary_calculation = []
    for s in info:
        salary = float(s[2])
        if salary >= minimum and salary <= maximum:
            salary_calculation.append(s)

    # Sort the list of employee information tuples by salary in descending order.
    sorted_salary = sorted(salary_calculation, key=employee_salary, reverse=True)
    print()

    # Check if any matches were found for the given salary range.
    if(len(salary_calculation)==0):
        print('No match found for the given minimum and maximum salary.')
    else:
        # Print the sorted list of employee information as a table.
        print(tabulate(sorted_salary, headers=["empName", "Designation", "Salary"]))


def employee_details_file():
    # create and empty list to store the employee details in a tuple
    list_of_tuples = []
    while True:
        # Get the file name from the user
        fileInput = input('Please enter the file that you wish to open: ', )
        
        try:
            # Open the file
            csvFile = open(fileInput)

            # Loop through each line of the file
            for items in csvFile:
                # Split the line into individual values and create a tuple
                empname, designation, salary = items.strip().split(',')
                newTuple = (empname, designation, salary)

                # Print the employee details
                print(f"Name: {empname}, Job title: {designation}, Salary: {salary}")

                # Add the tuple to the list
                list_of_tuples.append(newTuple)

            # Close the file and return the list of tuples
            csvFile.close()
            return list_of_tuples

        except FileNotFoundError:
            # If the file doesn't exist, print an error message and prompt the user to enter a valid file name
            print(f"Desired file does not exist '{fileInput}', Please enter a valid file name")

# This function calls the employee_details_file function 
def start():
    while True:
        employeeTuple = employee_details_file()
            # Prompt user to enter salary range and filter employee details based on range
        while True:
            minimum = float(input("Enter the minimum salary : "))
            maximum = float(input("Enter the maximum salary : "))

            employee_details(employeeTuple, minimum, maximum)

            # Ask the user if they want to continue with a new salary range or quit the program.
            quit_the_loop = input("Press 'y' to supply a new salary range or press 'n' to quit :")

            # If the user wants to continue, loop back to the top of the while loop.
            if quit_the_loop.lower() == 'y':
                continue 

            # If the user wants to quit, exit the function and print a message indicating the end of the program.
            elif quit_the_loop.lower() == 'n':
                print('End of the code')
                return          
start()

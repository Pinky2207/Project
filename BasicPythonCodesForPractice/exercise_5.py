import numpy as np

# This function calculates the overall mark for a course based on the student's exam mark, course mark, and the weight of the course mark in the overall mark calculation.
def overall_mark_calculation(exam_mark, course_mark, course_weight):

    # Calculate the weight of the exam in the overall mark calculation.
    exam_weight = 100 - course_weight

    # Calculate the contribution of the course mark to the overall mark.
    course_mark = (course_mark/100) * course_weight

    # Calculate the contribution of the exam mark to the overall mark.
    exam_mark = (exam_mark/100) * exam_weight

    # Calculate the overall mark by adding the contribution of the course mark and exam mark.
    overall_mark = exam_mark + course_mark

    return overall_mark

# This function takes in three arguments, coursework, examMark, and overallMark. It calculates the final grade of the student based on the given marks.
def grade_calculator(coursework,examMark, overallMark):

    # The function initializes an empty string 'grade'.
    grade = ''

    # If either coursework or examMark is less than 30, the function sets the grade to 'Fail'.
    if round(coursework)<30 or round(examMark) <30:
        # If overallMark is less than 40, the function sets the grade to 'Fail'.
        grade = 'Fail'

    # If the overall mark is less then 40 and then the function sets the grade to "Fail"
    elif round(overallMark) < 40:
        grade = 'Fail'

    # If the overall mark is less or equal to 49 then the function sets the grade to C
    elif 40<= round(overallMark) <=49:
        grade = 'Grade C'

    # If the overall mark is less than or equal to 69 then the function sets the grade to B
    elif 50<= round(overallMark) <= 69:
        grade = 'Grade B'

    # Final step, if the overall mark is greater than or equal to 70 then the function sets the grade to A
    elif round(overallMark)>= 70:
        grade ='Grade A'
    
    # Fianlly, the function would return the Grade based upon the inputs of student information
    return grade
    

def file():

    # Ask the user to enter the file which he/she wishes to open
    file_input = input("Enter the file that you wish to open: ")

    try:
        # Tries to open the file and reads the content 
        file_open = open(file_input, 'r')

    # If no such file found then it throws error 
    except FileNotFoundError:
        raise FileNotFoundError('Error: Unable to open the file')
    
    # Else it opens the files, reads the content and splits into two as number of students and coursework percent
    else:
        content = file_open.read().splitlines()
        first_line = content[0] # "5 30"
        num_of_students, coursework_percent = first_line.split(' ')

        # Initializes a NumPy array with the number of rows equal to the number of students and four columns.
        student_array = np.array([[0, 0.0, 0.0, 0.0]]*(int(num_of_students)))

        # Reads the remaining lines of the file, extracts the necessary data, and calculates the overall marks for each student.
        other_contents = content[1:]
        for index, lines in enumerate(other_contents):
            regNum, eMark, cMark = lines.strip().split(' ')
            overall = overall_mark_calculation(float(eMark), float(cMark), float(coursework_percent))
            data = [regNum, eMark, cMark, overall]
            
            student_array[index] = data
        
        print(student_array)
    
    # Creates a dictionary to keep track of the number of students in each grade category and the list of students who failed.
        studType = [('regNum', int), ('eMark', int), ('cMark', int), ('overall', int), ('Grade', 'U10')]
        s = np.array([], dtype=studType)
        student_dic = {'Grade A' : 0, 'Grade B' : 0, 'Grade C' : 0, 'Fail': 0, 'failedStudents': [] }

    # Uses a loop to calculate the grade for each student and stores the information in a NumPy array.
        for student_data in student_array:
            regNum, eMark, cMark, overall = student_data
            student_grade = grade_calculator(cMark, eMark, overall)
            student_tuple = (regNum, eMark, cMark, overall, student_grade)

            student_dic[student_grade] += 1
            if student_grade == 'Fail':
                student_dic['failedStudents'].append(int(regNum))

            arr = np.array([student_tuple], dtype=studType)
            s = np.append(s, arr, axis=0)


        # Sorts the NumPy array by overall marks.
        sorted_array = np.sort(s, order='overall')
        print(sorted_array)

        # Writes the sorted array to a CSV file named "studentResult.csv".
        with open("studentResult.csv", "w+") as open_file:
            print(sorted_array, file=open_file)
            
    # Prints the dictionary containing the number of students in each grade category and the list of failed students.
    print(student_dic)

file()




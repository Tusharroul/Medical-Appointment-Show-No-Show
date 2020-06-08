# Medical-Appointment-Show-No-Show

**Introduction to project**
No-show appointments, defined as an appointment in which the patient did not present for treatment or cancelled the same day as the appointment, are problematic for practices at all levels of the health care system. No-shows are a missed revenue opportunity which can’t be recaptured for the practice, and which contribute to both decreased patient and staff satisfaction
In this project, I am going to investigate a dataset of appointment records of Hospitals. In this dataset, the data includes some attributes of patients and state. And according to these attributes, whether the patient showed up to the appointments or not.
So Here the task is to analyse this dataset and will be focused on finding the condition which influence the patient to show or not show up to appointments
Basically, it’s Classification problem.

**Data Description**

PatientId: Identification of patients.

AppointmentID: Identification of each appointment

Gender: Gender of the patient
Scheduled Day: The day when the patient set up their appointment
Appointment Day: The day when the patient has to come to hospital to consult the doctor.
Age: Age of the patient
Neighborhood: Location of the patient
Scholarship: It indicates that whether the patient has some Health insurance or not
Hypertension: It indicates that whether the patient has hypertension or not.
Diabetes: It indicates that whether the patient has diabetes or not.
Alcoholism: It indicates that whether the patient is alcoholic or not.
Handicap: It indicates that whether the patient is handicapper or not.
SMS received: It indicates that whether the patient has received the SMS or not.
No-show: It indicates that whether the patient has showed up to their appointment or not.

**Modelling:**

**Logistic Regression:** The logistic model is used to model the
probability of a certain class or event existing such as pass/fail,
win/lose, alive/dead or healthy/sick. This can be extended to
model several classes of events such as determining whether an
image contains a cat, dog, lion, etc.

**Decision Tree:** A decision tree is a decision support tool that uses
a tree-like model of decisions and their possible consequences,
including chance event outcomes, resource costs, and utility. It is
one way to display an algorithm that only contains conditional
control statements.

**Random Forest:** Random forests or random decision forests are
an ensemble learning method for classification, regression and
other tasks that operate by constructing a multitude of decision
trees at training time and outputting the class that is the mode of
the classes or mean prediction of the individual trees

**Gradient boosting classifiers** are a group of machine learning
algorithms that combine many weak learning models together to
create a strong predictive model. Decision trees are usually used
when doing gradient boosting

**Conclusion:**
As a result, we can say that Logistic Regression is the best fit model with
Accuracy 72.91%, Precision Score 34.80%, Recall score 40.14% and F1
score 37.28%.


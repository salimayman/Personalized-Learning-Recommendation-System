# ================================================================
# PERSONALIZED LEARNING RECOMMENDATION SYSTEM
# Dataset: Open University Learning Analytics Dataset (OULAD)
# Language: Python
# Interface: Terminal Based
# ================================================================

import os
import json
import warnings
from pathlib import Path
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings("ignore")


# ================================================================
# STEP 1: PROJECT CONFIGURATION
# ================================================================

BASE_DIR = Path(__file__).resolve().parent

# OULAD DATASET FOLDER
DATASET_DIR = BASE_DIR / "dataset" / "OULAD"

# Files for saving progress and trained model
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

PROGRESS_FILE = DATA_DIR / "student_progress.json"
MODEL_FILE = MODEL_DIR / "oulad_random_forest.pkl"


# ================================================================
# STEP 2: COURSE AND SUBJECT CATALOG
# ================================================================

COURSE_SUBJECTS = {

    "BCA": [
        "Python Programming",
        "Java Programming",
        "C Programming",
        "C++ Programming",
        "Data Structures",
        "Database Management System",
        "Operating System",
        "Computer Networks",
        "Software Engineering",
        "Web Development",
        "Computer Organization",
        "Mathematics",
        "Artificial Intelligence",
        "Machine Learning",
        "Cyber Security"
    ],

    "B.Tech": [
        "Engineering Mathematics",
        "Engineering Physics",
        "Engineering Chemistry",
        "C Programming",
        "Python Programming",
        "Data Structures",
        "Design and Analysis of Algorithms",
        "Digital Electronics",
        "Computer Organization",
        "Database Management System",
        "Operating System",
        "Computer Networks",
        "Software Engineering",
        "Artificial Intelligence",
        "Machine Learning"
    ],

    "BBA": [
        "Principles of Management",
        "Financial Accounting",
        "Business Economics",
        "Marketing Management",
        "Human Resource Management",
        "Business Communication",
        "Business Mathematics",
        "Business Statistics",
        "Financial Management",
        "Organizational Behaviour",
        "Entrepreneurship",
        "Business Law",
        "Operations Management",
        "Strategic Management",
        "Digital Marketing"
    ],

    "MCA": [
        "Advanced Java",
        "Python Programming",
        "Data Structures",
        "Design and Analysis of Algorithms",
        "Database Management System",
        "Operating System",
        "Computer Networks",
        "Software Engineering",
        "Web Technologies",
        "Artificial Intelligence",
        "Machine Learning",
        "Cloud Computing",
        "Cyber Security",
        "Data Science",
        "Mobile Application Development"
    ]
}


# ================================================================
# STEP 3: SUBJECT TOPICS
# ================================================================

TOPIC_MAP = {

    "Python Programming": [
        "Python Basics",
        "Variables and Data Types",
        "Conditional Statements",
        "Loops",
        "Functions",
        "Lists and Tuples",
        "Dictionaries and Sets",
        "Object Oriented Programming",
        "File Handling",
        "Exception Handling"
    ],

    "Java Programming": [
        "Java Basics",
        "Variables and Data Types",
        "Control Statements",
        "Arrays",
        "Classes and Objects",
        "Inheritance",
        "Polymorphism",
        "Exception Handling",
        "Collections",
        "Multithreading"
    ],

    "C Programming": [
        "Introduction to C",
        "Variables and Data Types",
        "Operators",
        "Conditional Statements",
        "Loops",
        "Arrays",
        "Strings",
        "Functions",
        "Pointers",
        "Structures"
    ],

    "C++ Programming": [
        "C++ Basics",
        "Variables and Data Types",
        "Functions",
        "Arrays",
        "Pointers",
        "Classes and Objects",
        "Inheritance",
        "Polymorphism",
        "Encapsulation",
        "File Handling"
    ],

    "Data Structures": [
        "Arrays",
        "Linked Lists",
        "Stacks",
        "Queues",
        "Trees",
        "Binary Search Trees",
        "Graphs",
        "Hashing",
        "Searching Algorithms",
        "Sorting Algorithms"
    ],

    "Database Management System": [
        "Database Concepts",
        "ER Model",
        "Relational Model",
        "SQL Queries",
        "Normalization",
        "Primary and Foreign Keys",
        "Joins",
        "Transactions",
        "Concurrency Control",
        "Database Security"
    ],

    "Operating System": [
        "Introduction to OS",
        "Process Management",
        "CPU Scheduling",
        "Threads",
        "Process Synchronization",
        "Deadlocks",
        "Memory Management",
        "Virtual Memory",
        "File Systems",
        "Disk Scheduling"
    ],

    "Computer Networks": [
        "Network Fundamentals",
        "OSI Model",
        "TCP/IP Model",
        "IP Addressing",
        "Subnetting",
        "Routing",
        "Switching",
        "TCP and UDP",
        "DNS and HTTP",
        "Network Security"
    ],

    "Machine Learning": [
        "Introduction to ML",
        "Supervised Learning",
        "Unsupervised Learning",
        "Regression",
        "Classification",
        "Decision Trees",
        "Random Forest",
        "Clustering",
        "Model Evaluation",
        "Feature Engineering"
    ],

    "Artificial Intelligence": [
        "Introduction to AI",
        "Intelligent Agents",
        "Search Algorithms",
        "Breadth First Search",
        "Depth First Search",
        "Heuristic Search",
        "Knowledge Representation",
        "Expert Systems",
        "Neural Networks",
        "Natural Language Processing"
    ],

    "Software Engineering": [
        "Software Development Life Cycle",
        "Waterfall Model",
        "Agile Model",
        "Requirement Analysis",
        "System Design",
        "Software Testing",
        "Verification and Validation",
        "Software Maintenance",
        "Project Management",
        "Software Quality"
    ],

    "Web Development": [
        "HTML",
        "CSS",
        "JavaScript",
        "DOM Manipulation",
        "Responsive Design",
        "Frontend Development",
        "Backend Development",
        "APIs",
        "Database Connectivity",
        "Web Security"
    ]
}


def get_topics(subject):
    """Return topics for a selected subject."""

    if subject in TOPIC_MAP:
        return TOPIC_MAP[subject]

    return [
        f"Introduction to {subject}",
        f"Basic Concepts of {subject}",
        f"Fundamentals of {subject}",
        f"Important Terminologies of {subject}",
        f"Core Principles of {subject}",
        f"Applications of {subject}",
        f"Practical Examples of {subject}",
        f"Advanced Concepts of {subject}"
    ]


# ================================================================
# STEP 4: STUDENT INPUT
# ================================================================

def get_integer(prompt, minimum, maximum):

    while True:
        try:
            value = int(input(prompt))

            if minimum <= value <= maximum:
                return value

            print(f"Enter a value between {minimum} and {maximum}.")

        except ValueError:
            print("Please enter a valid number.")


def get_float(prompt, minimum, maximum):

    while True:
        try:
            value = float(input(prompt))

            if minimum <= value <= maximum:
                return value

            print(f"Enter a value between {minimum} and {maximum}.")

        except ValueError:
            print("Please enter a valid number.")


def collect_student_data():

    print("\n" + "=" * 65)
    print("PERSONALIZED LEARNING RECOMMENDATION SYSTEM")
    print("=" * 65)

    student_id = input("\nEnter your Student ID: ").strip()

    if not student_id:
        student_id = "DEFAULT_STUDENT"

    courses = list(COURSE_SUBJECTS.keys())

    print("\nAVAILABLE COURSES")

    for index, course in enumerate(courses, 1):
        print(f"{index}. {course}")

    course_choice = get_integer(
        "\nSelect your course: ",
        1,
        len(courses)
    )

    selected_course = courses[course_choice - 1]

    subjects = COURSE_SUBJECTS[selected_course]

    print(f"\nSelected Course: {selected_course}")
    print("\nAVAILABLE SUBJECTS")

    for index, subject in enumerate(subjects, 1):
        print(f"{index}. {subject}")

    print("\nSelect at least 6 subjects.")

    while True:

        try:
            choices = input(
                "\nEnter subject numbers separated by commas: "
            )

            selected_numbers = [
                int(x.strip()) for x in choices.split(",")
            ]

            if len(set(selected_numbers)) != len(selected_numbers):
                print("Duplicate subjects are not allowed.")
                continue

            if len(selected_numbers) < 6:
                print("Please select at least 6 subjects.")
                continue

            if any(
                number < 1 or number > len(subjects)
                for number in selected_numbers
            ):
                print("Invalid subject number.")
                continue

            break

        except ValueError:
            print("Enter valid subject numbers.")

    selected_subjects = [
        subjects[number - 1]
        for number in selected_numbers
    ]

    marks = {}

    print("\nENTER SUBJECT MARKS (OUT OF 100)")

    for subject in selected_subjects:

        marks[subject] = get_float(
            f"{subject}: ",
            0,
            100
        )

    study_hours = get_float(
        "\nDaily study hours: ",
        0.5,
        16
    )

    learning_methods = [
        "Video Tutorials",
        "Books and Notes",
        "Practical Exercises",
        "Mixed Learning"
    ]

    print("\nPREFERRED LEARNING METHOD")

    for index, method in enumerate(learning_methods, 1):
        print(f"{index}. {method}")

    method_choice = get_integer(
        "\nSelect learning method: ",
        1,
        len(learning_methods)
    )

    learning_method = learning_methods[method_choice - 1]

    return {
        "student_id": student_id,
        "course": selected_course,
        "marks": marks,
        "study_hours": study_hours,
        "learning_method": learning_method
    }


# ================================================================
# STEP 5: PERFORMANCE ANALYSIS
# ================================================================

def analyze_performance(marks):

    average = np.mean(list(marks.values()))

    weak_subjects = []
    subject_results = []

    print("\n" + "=" * 65)
    print("SUBJECT PERFORMANCE ANALYSIS")
    print("=" * 65)

    for subject, score in marks.items():

        if score >= 75:
            status = "Excellent"

        elif score >= 60:
            status = "Good"

        elif score >= 50:
            status = "Average"

        elif score >= 40:
            status = "Weak"

        else:
            status = "Very Weak"

        print(f"{subject:<40} {score:>6.1f}%  {status}")

        subject_results.append({
            "subject": subject,
            "marks": score,
            "status": status
        })

        if score < 60:
            weak_subjects.append(subject)

    weak_subjects.sort(key=lambda x: marks[x])

    print("\nOverall Average:", round(average, 2), "%")

    if weak_subjects:
        print("\nSubjects Requiring Improvement:")

        for index, subject in enumerate(weak_subjects, 1):
            print(
                f"{index}. {subject} - {marks[subject]}%"
            )

    else:
        print("\nGreat! No weak subjects identified.")

    return average, weak_subjects


# ================================================================
# STEP 6: PERSONALIZED RECOMMENDATIONS
# ================================================================

def generate_recommendations(
    weak_subjects,
    marks,
    learning_method
):

    print("\n" + "=" * 65)
    print("PERSONALIZED LEARNING RECOMMENDATIONS")
    print("=" * 65)

    if not weak_subjects:
        print("Maintain your performance with regular revision.")
        return

    for subject in weak_subjects:

        score = marks[subject]
        topics = get_topics(subject)

        print(f"\nSUBJECT: {subject}")
        print(f"CURRENT MARKS: {score}%")

        if score < 40:
            priority = "HIGH"
            recommended_topics = topics[:5]

        elif score < 50:
            priority = "HIGH"
            recommended_topics = topics[:4]

        else:
            priority = "MEDIUM"
            recommended_topics = topics[:3]

        print("Priority:", priority)

        print("\nRecommended Topics:")

        for topic in recommended_topics:

            print(f"\n  - {topic}")

            search_query = quote_plus(
                subject + " " + topic + " tutorial"
            )

            youtube_url = (
                "https://www.youtube.com/results?search_query="
                + search_query
            )

            google_url = (
                "https://www.google.com/search?q="
                + quote_plus(subject + " " + topic + " notes")
            )

            print("    Video Resource:", youtube_url)
            print("    Notes Resource:", google_url)

        print("\nHOW TO STUDY:")

        if learning_method == "Video Tutorials":
            print("Watch video lectures and make short notes.")

        elif learning_method == "Books and Notes":
            print("Read textbook concepts and revise important notes.")

        elif learning_method == "Practical Exercises":
            print("Solve practical problems and practice questions.")

        else:
            print("Combine videos, notes and practical exercises.")


# ================================================================
# STEP 7: WEEKLY STUDY PLANNER
# ================================================================

def create_study_plan(
    weak_subjects,
    marks,
    study_hours
):

    print("\n" + "=" * 65)
    print("PERSONALIZED 7-DAY STUDY PLANNER")
    print("=" * 65)

    days = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]

    if not weak_subjects:
        subjects = list(marks.keys())
    else:
        subjects = weak_subjects

    for index, day in enumerate(days):

        subject = subjects[index % len(subjects)]

        print(f"\n{day}")
        print(f"  Main Subject: {subject}")
        print(f"  Study Time: {study_hours} hours")

        if marks[subject] < 40:
            print("  Task: Learn basic concepts and solve easy questions.")

        elif marks[subject] < 60:
            print("  Task: Revise important topics and practice questions.")

        else:
            print("  Task: Revision, advanced topics and mock tests.")

    print("\nWeekly Study Target:")
    print(round(study_hours * 7, 2), "hours")


# ================================================================
# STEP 8: IMPROVEMENT PLANNING
# ================================================================

def improvement_scenario(average, study_hours):

    print("\n" + "=" * 65)
    print("ACADEMIC IMPROVEMENT SCENARIO")
    print("=" * 65)

    if average < 40:
        improvement = 12

    elif average < 60:
        improvement = 8

    elif average < 75:
        improvement = 5

    else:
        improvement = 2

    # Study hours influence the planning scenario.
    improvement *= min(study_hours / 3, 1.5)

    projected_average = min(
        100,
        average + improvement
    )

    print("Current Average:", round(average, 2), "%")
    print("Potential Planning Scenario:", round(projected_average, 2), "%")
    print("Possible Difference:", round(projected_average - average, 2), "points")

    print(
        "\nNOTE: This is a rule-based planning scenario, "
        "not a validated ML prediction or guarantee."
    )


# ================================================================
# STEP 9: PROGRESS TRACKING
# ================================================================

def save_progress(student_id, course, marks, average):

    if PROGRESS_FILE.exists():

        try:
            with open(PROGRESS_FILE, "r") as file:
                progress = json.load(file)

        except (json.JSONDecodeError, OSError):
            progress = {}

    else:
        progress = {}

    if student_id not in progress:
        progress[student_id] = []

    previous_records = progress[student_id]

    if previous_records:

        previous = previous_records[-1]

        print("\n" + "=" * 65)
        print("PREVIOUS PERFORMANCE COMPARISON")
        print("=" * 65)

        print(
            "Previous Average:",
            previous["average"]
        )

        print(
            "Current Average:",
            round(average, 2)
        )

        difference = average - previous["average"]

        if difference > 0:
            print("Performance Improved by:", round(difference, 2))

        elif difference < 0:
            print("Performance Decreased by:", round(abs(difference), 2))

        else:
            print("Performance Remained the Same.")

        print("\nSubject-wise Comparison:")

        for subject, score in marks.items():

            old_score = previous["marks"].get(subject)

            if old_score is not None:

                change = score - old_score

                print(
                    f"{subject}: Previous {old_score}, "
                    f"Current {score}, Change {change:+.1f}"
                )

    progress[student_id].append({
        "course": course,
        "marks": marks,
        "average": round(float(average), 2)
    })

    with open(PROGRESS_FILE, "w") as file:
        json.dump(progress, file, indent=4)

    print("\nStudent progress saved successfully.")


# ================================================================
# STEP 10: OULAD DATASET INTEGRATION
# ================================================================
#
# IMPORTANT:
# THIS IS THE MAIN SECTION WHERE OULAD CSV FILES ARE USED.
#
# Dataset folder:
# dataset/OULAD/
#
# CSV FILES USED:
#
# 1. studentInfo.csv
#    Student profile and final_result.
#
# 2. assessments.csv
#    Assessment information.
#
# 3. studentAssessment.csv
#    Actual student assessment scores.
#
# 4. studentVle.csv
#    Student learning activity and clicks.
#
# 5. vle.csv
#    Learning resource information.
#
# 6. courses.csv
#    Course module and presentation details.
#
# 7. studentRegistration.csv
#    Student registration information (optional).
#
# All available data is merged to create historical
# student-level features for Random Forest classification.
#
# The target variable is final_result from studentInfo.csv.
#
# OULAD does not contain BCA subject-wise marks.
# BCA subject performance is analyzed separately above.
# ================================================================


# ------------------------------------------------
# STEP 10.1: LOAD OULAD CSV FILES
# ------------------------------------------------

def load_oulad_dataset():

    print("\n" + "=" * 65)
    print("OULAD DATASET: CSV LOADING")
    print("=" * 65)

    required_files = [
        "studentInfo.csv",
        "assessments.csv",
        "studentAssessment.csv",
        "studentVle.csv",
        "vle.csv",
        "courses.csv"
    ]

    data = {}

    if not DATASET_DIR.exists():

        print("\nOULAD dataset folder not found.")
        print("Expected location:", DATASET_DIR)

        return None

    # Read all required CSV files.
    for filename in required_files:

        filepath = DATASET_DIR / filename

        if not filepath.exists():

            print("Missing CSV file:", filename)
            return None

        try:

            # CSV DATA IS READ HERE.
            data[filename.replace(".csv", "")] = pd.read_csv(filepath)

            print("Loaded:", filename)

        except Exception as error:

            print("Error loading", filename, ":", error)
            return None

    # Registration CSV is optional.
    registration_file = DATASET_DIR / "studentRegistration.csv"

    if registration_file.exists():

        data["studentRegistration"] = pd.read_csv(
            registration_file
        )

        print("Loaded: studentRegistration.csv")

    return data


# ------------------------------------------------
# STEP 10.2: DATA PREPROCESSING AND MERGING
# ------------------------------------------------

def preprocess_oulad_data(data):

    print("\nOULAD DATA PREPROCESSING STARTED")

    student_info = data["studentInfo"].copy()

    assessments = data["assessments"].copy()

    student_assessment = data["studentAssessment"].copy()

    student_vle = data["studentVle"].copy()

    vle = data["vle"].copy()

    courses = data["courses"].copy()

    # ------------------------------------------------
    # A. STUDENT ASSESSMENT DATA
    #
    # assessments.csv + studentAssessment.csv
    #
    # Merge using id_assessment.
    # Calculate average assessment score and count.
    # ------------------------------------------------

    assessment_data = student_assessment.merge(
        assessments,
        on="id_assessment",
        how="left"
    )

    assessment_data["score"] = pd.to_numeric(
        assessment_data["score"],
        errors="coerce"
    )

    assessment_data["date_submitted"] = pd.to_numeric(
        assessment_data["date_submitted"],
        errors="coerce"
    )

    assessment_data["date"] = pd.to_numeric(
        assessment_data["date"],
        errors="coerce"
    )

    assessment_data["late_submission"] = (
        assessment_data["date_submitted"]
        > assessment_data["date"]
    ).fillna(False).astype(int)

    assessment_features = assessment_data.groupby(
        ["id_student", "code_module", "code_presentation"]
    ).agg(
        average_assessment_score=("score", "mean"),
        assessment_count=("score", "count"),
        late_submission_count=("late_submission", "sum")
    ).reset_index()

    # ------------------------------------------------
    # B. STUDENT VLE ACTIVITY
    #
    # studentVle.csv + vle.csv
    #
    # Merge using id_site.
    # Calculate total clicks and active resources.
    # ------------------------------------------------

    vle_data = student_vle.merge(
        vle[["id_site", "code_module", "code_presentation"]],
        on="id_site",
        how="left",
        suffixes=("", "_resource")
    )

    # Use student activity's course columns when available.
    for column in ["code_module", "code_presentation"]:

        resource_column = column + "_resource"

        if resource_column in vle_data.columns:

            vle_data[column] = vle_data[column].fillna(
                vle_data[resource_column]
            )

    vle_features = vle_data.groupby(
        ["id_student", "code_module", "code_presentation"]
    ).agg(
        total_clicks=("sum_click", "sum"),
        active_days=("date", "nunique"),
        unique_resources=("id_site", "nunique")
    ).reset_index()

    # ------------------------------------------------
    # C. MERGE STUDENT PROFILE WITH COURSE DATA
    #
    # studentInfo.csv + courses.csv
    #
    # Merge using code_module and code_presentation.
    # ------------------------------------------------

    course_columns = [
        column for column in [
            "code_module",
            "code_presentation",
            "module_presentation_length"
        ]
        if column in courses.columns
    ]

    final_data = student_info.merge(
        courses[course_columns],
        on=["code_module", "code_presentation"],
        how="left"
    )

    # ------------------------------------------------
    # D. ADD ASSESSMENT FEATURES
    # ------------------------------------------------

    final_data = final_data.merge(
        assessment_features,
        on=["id_student", "code_module", "code_presentation"],
        how="left"
    )

    # ------------------------------------------------
    # E. ADD VLE ACTIVITY FEATURES
    # ------------------------------------------------

    final_data = final_data.merge(
        vle_features,
        on=["id_student", "code_module", "code_presentation"],
        how="left"
    )

    # ------------------------------------------------
    # F. ADD REGISTRATION INFORMATION (IF AVAILABLE)
    # ------------------------------------------------

    if "studentRegistration" in data:

        registration = data["studentRegistration"].copy()

        registration_columns = [
            column for column in [
                "id_student",
                "code_module",
                "code_presentation",
                "date_registration",
                "date_unregistration"
            ]
            if column in registration.columns
        ]

        registration = registration[registration_columns]

        final_data = final_data.merge(
            registration,
            on=[
                "id_student",
                "code_module",
                "code_presentation"
            ],
            how="left"
        )

    # ------------------------------------------------
    # G. CLEANING MISSING VALUES
    # ------------------------------------------------

    numerical_columns = [
        "average_assessment_score",
        "assessment_count",
        "late_submission_count",
        "total_clicks",
        "active_days",
        "unique_resources"
    ]

    for column in numerical_columns:

        if column in final_data.columns:
            final_data[column] = final_data[column].fillna(0)

    # Remove records without target values.
    final_data = final_data.dropna(
        subset=["final_result"]
    )

    print("OULAD preprocessing completed.")
    print("Total student records:", len(final_data))

    return final_data


# ------------------------------------------------
# STEP 10.3: RANDOM FOREST MODEL TRAINING
# ------------------------------------------------

def train_oulad_model(final_data):

    print("\n" + "=" * 65)
    print("OULAD RANDOM FOREST MODEL TRAINING")
    print("=" * 65)

    # Target variable from studentInfo.csv.
    target = "final_result"

    # Exclude identifiers and target from input features.
    excluded_columns = [
        "id_student",
        "final_result"
    ]

    X = final_data.drop(
        columns=excluded_columns,
        errors="ignore"
    )

    y = final_data[target]

    # Identify numeric and categorical columns.
    numerical_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    # Numerical preprocessing.
    numerical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ])

    # Categorical preprocessing.
    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ])

    # Combine both preprocessing pipelines.
    preprocessor = ColumnTransformer([
        (
            "numeric",
            numerical_pipeline,
            numerical_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ])

    # Random Forest classifier.
    random_forest = RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    # Complete ML pipeline.
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", random_forest)
    ])

    # Check if enough samples exist for a meaningful split.
    if len(final_data) < 10:
        print("Not enough records for model evaluation.")
        return

    # Stratification is used only when every class has
    # at least two records.
    class_counts = y.value_counts()

    stratify_target = (
        y if len(class_counts) > 1
        and class_counts.min() >= 2
        else None
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=stratify_target
    )

    # Train model on OULAD historical data.
    model.fit(X_train, y_train)

    # Evaluate model on test data.
    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("\nMODEL EVALUATION")
    print("Accuracy:", round(accuracy * 100, 2), "%")

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    # Save trained model.
    joblib.dump(model, MODEL_FILE)

    print("Trained model saved at:", MODEL_FILE)

    print(
        "\nIMPORTANT: This model classifies historical OULAD "
        "final-result patterns. It is not a validated forecast "
        "of the student's future BCA subject marks."
    )


# ------------------------------------------------
# STEP 10.4: RUN COMPLETE OULAD PIPELINE
# ------------------------------------------------

def run_oulad_analysis():

    print("\n" + "=" * 65)
    print("OULAD DATASET ANALYSIS")
    print("=" * 65)

    # STEP A: Load OULAD CSV files.
    dataset = load_oulad_dataset()

    if dataset is None:

        print("\nOULAD analysis skipped.")
        print("Please check your CSV files and dataset folder.")

        return

    # STEP B: Preprocess and merge CSV data.
    processed_data = preprocess_oulad_data(dataset)

    if processed_data is None or processed_data.empty:

        print("No valid OULAD records available.")
        return

    # STEP C: Train and evaluate Random Forest.
    train_oulad_model(processed_data)


# ================================================================
# STEP 11: MAIN PROGRAM
# ================================================================

def main():

    # Collect student information.
    student_data = collect_student_data()

    marks = student_data["marks"]

    # Analyze student performance.
    average, weak_subjects = analyze_performance(marks)

    # Generate personalized recommendations.
    generate_recommendations(
        weak_subjects,
        marks,
        student_data["learning_method"]
    )

    # Generate weekly planner.
    create_study_plan(
        weak_subjects,
        marks,
        student_data["study_hours"]
    )

    # Show improvement planning scenario.
    improvement_scenario(
        average,
        student_data["study_hours"]
    )

    # Save and compare student progress.
    save_progress(
        student_data["student_id"],
        student_data["course"],
        marks,
        average
    )

    # Load OULAD CSV files and train Random Forest.
    run_oulad_analysis()

    print("\n" + "=" * 65)
    print("PERSONALIZED LEARNING ANALYSIS COMPLETED")
    print("=" * 65)


# ================================================================
# PROGRAM ENTRY POINT
# ================================================================

if __name__ == "__main__":
    main()
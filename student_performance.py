"""
Student Performance Prediction System
A complete ML system for predicting student marks and pass/fail status

Author: [Your Name]
Version: 1.0
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, precision_score, recall_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class StudentPerformanceSystem:
    """Main class for student performance prediction"""
    
    def __init__(self):
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train_marks = None
        self.y_test_marks = None
        self.y_train_pass = None
        self.y_test_pass = None
        self.marks_model = None
        self.pass_fail_model = None
    
    def print_header(self, text):
        """Prints formatted section headers"""
        print("\n" + "="*60)
        print(f"  {text}")
        print("="*60)
    
    def create_dataset(self):
        """Creates student dataset with user input or generates sample data"""
        self.print_header("📝 DATA ENTRY MODULE")
        
        print("\nChoose data entry method:")
        print("1. Manual Entry (Enter student data)")
        print("2. Generate Sample Data (Auto-generate 200 students)")
        
        choice = input("\nEnter your choice (1 or 2): ").strip()
        
        if choice == '1':
            self.manual_data_entry()
        else:
            self.generate_sample_data()
    
    def manual_data_entry(self):
        """Allows manual entry of student data"""
        print("\n📊 Manual Data Entry")
        print("-" * 60)
        
        students = []
        
        while True:
            print(f"\n--- Student #{len(students) + 1} ---")
            
            try:
                study_hours = float(input("Study Hours per week (0-15): "))
                attendance = float(input("Attendance Percentage (0-100): "))
                previous_score = float(input("Previous Semester Score (0-100): "))
                assignments = int(input("Assignments Completed (0-10): "))
                final_marks = float(input("Final Marks (0-100): "))
                
                students.append({
                    'StudyHours': study_hours,
                    'AttendancePercentage': attendance,
                    'PreviousScore': previous_score,
                    'AssignmentsCompleted': assignments,
                    'FinalMarks': final_marks
                })
                
                print(f"✓ Student #{len(students)} added successfully!")
                
                more = input("\nAdd another student? (y/n): ").strip().lower()
                if more != 'y':
                    break
                    
            except ValueError:
                print("❌ Invalid input! Please enter numeric values.")
                continue
        
        self.df = pd.DataFrame(students)
        self.df['Passed'] = self.df['FinalMarks'] >= 50
        
        # Save to CSV
        self.df.to_csv('student_data.csv', index=False)
        print(f"\n✓ Dataset created with {len(students)} students")
        print("✓ Saved to 'student_data.csv'")
    
    def generate_sample_data(self):
        """Generates realistic sample data with correlations"""
        print("\n🔄 Generating sample dataset...")
        
        np.random.seed(42)
        n_samples = 200
        
        # Generate features with realistic distributions
        study_hours = np.random.uniform(1, 12, n_samples)
        attendance = np.random.uniform(45, 100, n_samples)
        previous_score = np.random.uniform(25, 95, n_samples)
        assignments = np.random.randint(0, 11, n_samples)
        
        # Generate final marks with correlations
        final_marks = (
            10 + 
            study_hours * 3.5 + 
            attendance * 0.35 + 
            previous_score * 0.38 + 
            assignments * 2.2 + 
            np.random.normal(0, 6, n_samples)
        )
        
        # Clip marks between 0 and 100
        final_marks = np.clip(final_marks, 0, 100)
        
        # Create DataFrame
        self.df = pd.DataFrame({
            'StudyHours': study_hours,
            'AttendancePercentage': attendance,
            'PreviousScore': previous_score,
            'AssignmentsCompleted': assignments,
            'FinalMarks': final_marks
        })
        
        self.df['Passed'] = self.df['FinalMarks'] >= 50
        
        # Save to CSV
        self.df.to_csv('student_data.csv', index=False)
        print(f"✓ Generated {n_samples} student records")
        print("✓ Saved to 'student_data.csv'")
    
    def load_data(self):
        """Loads data from CSV file"""
        self.print_header("📂 DATA LOADING")
        
        try:
            self.df = pd.read_csv('student_data.csv')
            print(f"✓ Loaded {len(self.df)} student records from 'student_data.csv'")
            
            # Ensure Passed column exists
            if 'Passed' not in self.df.columns:
                self.df['Passed'] = self.df['FinalMarks'] >= 50
                
            print("\n📋 First 5 records:")
            print(self.df.head())
            
        except FileNotFoundError:
            print("❌ File 'student_data.csv' not found!")
            print("Creating new dataset...")
            self.create_dataset()
    
    def analyze_data(self):
        """Performs exploratory data analysis"""
        self.print_header("📊 EXPLORATORY DATA ANALYSIS")
        
        # Basic Statistics
        print("\n📈 Dataset Statistics:")
        print("-" * 60)
        print(f"Total Students: {len(self.df)}")
        print(f"\nFeature Averages:")
        print(f"  Study Hours: {self.df['StudyHours'].mean():.2f} hours/week")
        print(f"  Attendance: {self.df['AttendancePercentage'].mean():.2f}%")
        print(f"  Previous Score: {self.df['PreviousScore'].mean():.2f}")
        print(f"  Assignments Completed: {self.df['AssignmentsCompleted'].mean():.2f}")
        print(f"  Final Marks: {self.df['FinalMarks'].mean():.2f}")
        
        print(f"\n📊 Marks Distribution:")
        print(f"  Minimum: {self.df['FinalMarks'].min():.2f}")
        print(f"  Maximum: {self.df['FinalMarks'].max():.2f}")
        print(f"  Median: {self.df['FinalMarks'].median():.2f}")
        print(f"  Std Dev: {self.df['FinalMarks'].std():.2f}")
        
        # Pass/Fail Statistics
        pass_count = self.df['Passed'].sum()
        pass_rate = (pass_count / len(self.df)) * 100
        print(f"\n✅ Pass/Fail Statistics:")
        print(f"  Passed: {pass_count} students ({pass_rate:.2f}%)")
        print(f"  Failed: {len(self.df) - pass_count} students ({100-pass_rate:.2f}%)")
        
        # Correlation Analysis
        print(f"\n🔗 Feature Correlations with Final Marks:")
        print("-" * 60)
        features = ['StudyHours', 'AttendancePercentage', 'PreviousScore', 'AssignmentsCompleted']
        correlations = self.df[features].corrwith(self.df['FinalMarks']).sort_values(ascending=False)
        
        for feature, corr in correlations.items():
            strength = self.get_correlation_strength(corr)
            print(f"  {feature:25s}: {corr:.3f} {strength}")
        
        # Create visualizations
        self.create_visualizations()
    
    def get_correlation_strength(self, corr):
        """Returns correlation strength label"""
        abs_corr = abs(corr)
        if abs_corr >= 0.7:
            return "(Strong)"
        elif abs_corr >= 0.4:
            return "(Moderate)"
        elif abs_corr >= 0.2:
            return "(Weak)"
        else:
            return "(Very Weak)"
    
    def create_visualizations(self):
        """Creates data visualizations"""
        print("\n📊 Creating visualizations...")
        
        # Set style
        sns.set_style("whitegrid")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Student Performance Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Marks Distribution
        axes[0, 0].hist(self.df['FinalMarks'], bins=20, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Final Marks Distribution')
        axes[0, 0].set_xlabel('Marks')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(50, color='red', linestyle='--', label='Pass Threshold')
        axes[0, 0].legend()
        
        # 2. Study Hours vs Marks
        axes[0, 1].scatter(self.df['StudyHours'], self.df['FinalMarks'], alpha=0.6, color='green')
        axes[0, 1].set_title('Study Hours vs Final Marks')
        axes[0, 1].set_xlabel('Study Hours (per week)')
        axes[0, 1].set_ylabel('Final Marks')
        
        # 3. Attendance vs Marks
        axes[0, 2].scatter(self.df['AttendancePercentage'], self.df['FinalMarks'], alpha=0.6, color='orange')
        axes[0, 2].set_title('Attendance vs Final Marks')
        axes[0, 2].set_xlabel('Attendance %')
        axes[0, 2].set_ylabel('Final Marks')
        
        # 4. Previous Score vs Marks
        axes[1, 0].scatter(self.df['PreviousScore'], self.df['FinalMarks'], alpha=0.6, color='purple')
        axes[1, 0].set_title('Previous Score vs Final Marks')
        axes[1, 0].set_xlabel('Previous Score')
        axes[1, 0].set_ylabel('Final Marks')
        
        # 5. Pass/Fail Distribution
        pass_counts = self.df['Passed'].value_counts()
        # Create labels based on what's actually in the data
        pie_labels = []
        pie_values = []
        pie_colors = []
        
        if False in pass_counts.index:
            pie_labels.append('Failed')
            pie_values.append(pass_counts[False])
            pie_colors.append('#ff6b6b')
        if True in pass_counts.index:
            pie_labels.append('Passed')
            pie_values.append(pass_counts[True])
            pie_colors.append('#51cf66')
        
        axes[1, 1].pie(pie_values, labels=pie_labels, autopct='%1.1f%%', 
                       colors=pie_colors, startangle=90)
        axes[1, 1].set_title('Pass/Fail Distribution')
        
        # 6. Correlation Heatmap
        features = ['StudyHours', 'AttendancePercentage', 'PreviousScore', 'AssignmentsCompleted', 'FinalMarks']
        corr_matrix = self.df[features].corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                    square=True, ax=axes[1, 2], cbar_kws={'shrink': 0.8})
        axes[1, 2].set_title('Feature Correlation Heatmap')
        
        plt.tight_layout()
        plt.savefig('analysis_dashboard.png', dpi=300, bbox_inches='tight')
        print("✓ Visualizations saved as 'analysis_dashboard.png'")
        plt.show()
    
    def prepare_data(self):
        """Prepares data for model training"""
        self.print_header("🔧 DATA PREPARATION")
        
        # Feature columns
        features = ['StudyHours', 'AttendancePercentage', 'PreviousScore', 'AssignmentsCompleted']
        
        X = self.df[features]
        y_marks = self.df['FinalMarks']
        y_pass = self.df['Passed'].astype(int)
        
        # Train-test split (80-20)
        self.X_train, self.X_test, self.y_train_marks, self.y_test_marks = train_test_split(
            X, y_marks, test_size=0.2, random_state=42
        )
        
        _, _, self.y_train_pass, self.y_test_pass = train_test_split(
            X, y_pass, test_size=0.2, random_state=42
        )
        
        print(f"✓ Training set: {len(self.X_train)} students (80%)")
        print(f"✓ Test set: {len(self.X_test)} students (20%)")
        print(f"\nFeatures used: {', '.join(features)}")
    
    def train_models(self):
        """Trains ML models"""
        self.print_header("🤖 MODEL TRAINING")
        
        # Train Linear Regression for Marks Prediction
        print("\n1️⃣ Training Linear Regression (Marks Prediction)...")
        self.marks_model = LinearRegression()
        self.marks_model.fit(self.X_train, self.y_train_marks)
        print("✓ Linear Regression model trained successfully")
        
        # Display feature importance (coefficients)
        print("\n   Feature Coefficients:")
        for feature, coef in zip(self.X_train.columns, self.marks_model.coef_):
            print(f"   {feature:25s}: {coef:8.3f}")
        print(f"   Intercept: {self.marks_model.intercept_:.3f}")
        
        # Train Logistic Regression for Pass/Fail Classification
        print("\n2️⃣ Training Logistic Regression (Pass/Fail Classification)...")
        
        # Check if we have both classes in training data
        unique_classes = self.y_train_pass.unique()
        if len(unique_classes) < 2:
            print("⚠️  Warning: Training data contains only one class!")
            if unique_classes[0] == 1:
                print("   All students in training set passed (marks >= 50)")
            else:
                print("   All students in training set failed (marks < 50)")
            print("   Skipping Pass/Fail model training.")
            print("   Tip: Add students with diverse marks (both < 50 and >= 50) for classification.")
            self.pass_fail_model = None
        else:
            self.pass_fail_model = LogisticRegression(max_iter=1000, random_state=42)
            self.pass_fail_model.fit(self.X_train, self.y_train_pass)
            print("✓ Logistic Regression model trained successfully")
    
    def evaluate_models(self):
        """Evaluates model performance"""
        self.print_header("📈 MODEL EVALUATION")
        
        # Evaluate Marks Prediction Model
        print("\n1️⃣ Marks Prediction Model (Linear Regression)")
        print("-" * 60)
        
        y_pred_marks = self.marks_model.predict(self.X_test)
        
        rmse = np.sqrt(mean_squared_error(self.y_test_marks, y_pred_marks))
        mae = mean_absolute_error(self.y_test_marks, y_pred_marks)
        r2_score = self.marks_model.score(self.X_test, self.y_test_marks)
        
        print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
        print(f"Mean Absolute Error (MAE): {mae:.2f}")
        print(f"R² Score: {r2_score:.3f}")
        
        # Evaluate Pass/Fail Classification Model (if trained)
        if self.pass_fail_model is not None:
            print("\n2️⃣ Pass/Fail Classification Model (Logistic Regression)")
            print("-" * 60)
            
            y_pred_pass = self.pass_fail_model.predict(self.X_test)
            
            accuracy = accuracy_score(self.y_test_pass, y_pred_pass)
            precision = precision_score(self.y_test_pass, y_pred_pass, zero_division=0)
            recall = recall_score(self.y_test_pass, y_pred_pass, zero_division=0)
            
            print(f"Accuracy: {accuracy*100:.2f}%")
            print(f"Precision: {precision*100:.2f}%")
            print(f"Recall: {recall*100:.2f}%")
            
            # Confusion Matrix
            cm = confusion_matrix(self.y_test_pass, y_pred_pass)
            print(f"\nConfusion Matrix:")
            print(f"                 Predicted")
            print(f"               Fail    Pass")
            print(f"Actual Fail    {cm[0,0]:4d}    {cm[0,1]:4d}")
            print(f"       Pass    {cm[1,0]:4d}    {cm[1,1]:4d}")
            
            # Create evaluation visualizations
            self.create_evaluation_plots(y_pred_marks, y_pred_pass)
        else:
            print("\n2️⃣ Pass/Fail Classification Model")
            print("-" * 60)
            print("⚠️  Model not trained (insufficient class diversity)")
            # Create evaluation plots for marks only
            self.create_evaluation_plots(y_pred_marks, None)
    
    def create_evaluation_plots(self, y_pred_marks, y_pred_pass):
        """Creates model evaluation visualizations"""
        print("\n📊 Creating evaluation plots...")
        
        if y_pred_pass is not None:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle('Model Evaluation Results', fontsize=14, fontweight='bold')
            
            # Actual vs Predicted Marks
            axes[0].scatter(self.y_test_marks, y_pred_marks, alpha=0.6, color='blue')
            axes[0].plot([0, 100], [0, 100], 'r--', linewidth=2, label='Perfect Prediction')
            axes[0].set_xlabel('Actual Marks')
            axes[0].set_ylabel('Predicted Marks')
            axes[0].set_title('Marks Prediction: Actual vs Predicted')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Confusion Matrix Heatmap
            cm = confusion_matrix(self.y_test_pass, y_pred_pass)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['Fail', 'Pass'], 
                        yticklabels=['Fail', 'Pass'],
                        ax=axes[1], cbar_kws={'label': 'Count'})
            axes[1].set_xlabel('Predicted')
            axes[1].set_ylabel('Actual')
            axes[1].set_title('Pass/Fail Classification: Confusion Matrix')
        else:
            # Only marks prediction plot
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            fig.suptitle('Model Evaluation Results', fontsize=14, fontweight='bold')
            
            ax.scatter(self.y_test_marks, y_pred_marks, alpha=0.6, color='blue')
            ax.plot([0, 100], [0, 100], 'r--', linewidth=2, label='Perfect Prediction')
            ax.set_xlabel('Actual Marks')
            ax.set_ylabel('Predicted Marks')
            ax.set_title('Marks Prediction: Actual vs Predicted')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
        print("✓ Evaluation plots saved as 'model_evaluation.png'")
        plt.show()
    
    def predict_new_students(self):
        """Makes predictions for new students"""
        self.print_header("🎯 PREDICTIONS FOR NEW STUDENTS")
        
        print("\nChoose prediction method:")
        print("1. Enter new student data manually")
        print("2. Use sample predictions")
        
        choice = input("\nEnter your choice (1 or 2): ").strip()
        
        if choice == '1':
            self.manual_prediction()
        else:
            self.sample_predictions()
    
    def manual_prediction(self):
        """Allows manual prediction for new students"""
        while True:
            print("\n" + "-"*60)
            print("Enter new student details:")
            
            try:
                study = float(input("Study Hours per week (0-15): "))
                attend = float(input("Attendance Percentage (0-100): "))
                prev = float(input("Previous Semester Score (0-100): "))
                assign = int(input("Assignments Completed (0-10): "))
                
                # Create input array
                student_data = np.array([[study, attend, prev, assign]])
                
                # Make predictions
                predicted_marks = self.marks_model.predict(student_data)[0]
                
                # Display results
                print("\n" + "="*60)
                print("📊 PREDICTION RESULTS")
                print("="*60)
                print(f"Study Hours: {study:.1f} | Attendance: {attend:.0f}% | Previous: {prev:.0f} | Assignments: {assign}")
                print(f"\n🎯 Predicted Final Marks: {predicted_marks:.1f}/100")
                
                # Only predict pass/fail if model is trained
                if self.pass_fail_model is not None:
                    predicted_pass = self.pass_fail_model.predict(student_data)[0]
                    print(f"📝 Prediction: {'PASS ✓' if predicted_pass == 1 else 'FAIL ✗'}")
                else:
                    # Use simple threshold
                    print(f"📝 Prediction: {'PASS ✓' if predicted_marks >= 50 else 'FAIL ✗'} (based on 50% threshold)")
                
                print("="*60)
                
            except ValueError:
                print("❌ Invalid input! Please enter numeric values.")
                continue
            
            more = input("\nPredict for another student? (y/n): ").strip().lower()
            if more != 'y':
                break
    
    def sample_predictions(self):
        """Shows sample predictions"""
        print("\n" + "-"*60)
        print("Sample Predictions:")
        print("-"*60)
        
        # Sample students
        samples = [
            {"name": "Excellent Student", "data": [10, 95, 90, 10]},
            {"name": "Good Student", "data": [8, 85, 75, 8]},
            {"name": "Average Student", "data": [6, 70, 65, 6]},
            {"name": "Below Average", "data": [4, 60, 50, 4]},
            {"name": "At-Risk Student", "data": [2, 45, 35, 2]}
        ]
        
        for sample in samples:
            student_data = np.array([sample['data']])
            predicted_marks = self.marks_model.predict(student_data)[0]
            
            print(f"\n{sample['name']:20s}")
            print(f"  Study: {sample['data'][0]}h | Attend: {sample['data'][1]}% | Prev: {sample['data'][2]} | Assign: {sample['data'][3]}")
            print(f"  → Predicted Marks: {predicted_marks:.1f}/100")
            
            # Only use pass/fail model if it exists
            if self.pass_fail_model is not None:
                predicted_pass = self.pass_fail_model.predict(student_data)[0]
                print(f"  → Result: {'PASS ✓' if predicted_pass == 1 else 'FAIL ✗'}")
            else:
                print(f"  → Result: {'PASS ✓' if predicted_marks >= 50 else 'FAIL ✗'} (50% threshold)")
    
    def run(self):
        """Runs the complete prediction system"""
        print("\n" + "╔" + "="*58 + "╗")
        print("║" + " "*10 + "STUDENT PERFORMANCE PREDICTION SYSTEM" + " "*11 + "║")
        print("║" + " "*25 + "Version 1.0" + " "*22 + "║")
        print("╚" + "="*58 + "╝")
        
        # Step 1: Create or Load Data
        print("\n📋 Do you want to:")
        print("1. Create new dataset")
        print("2. Load existing dataset (student_data.csv)")
        
        choice = input("\nEnter your choice (1 or 2): ").strip()
        
        if choice == '1':
            self.create_dataset()
        else:
            self.load_data()
        
        # Step 2: Analyze Data
        self.analyze_data()
        
        # Step 3: Prepare Data
        self.prepare_data()
        
        # Step 4: Train Models
        self.train_models()
        
        # Step 5: Evaluate Models
        self.evaluate_models()
        
        # Step 6: Make Predictions
        self.predict_new_students()
        
        print("\n" + "╔" + "="*58 + "╗")
        print("║" + " "*10 + "SYSTEM COMPLETED SUCCESSFULLY!" + " "*17 + "║")
        print("╚" + "="*58 + "╝\n")
        
        print("📁 Files generated:")
        print("  - student_data.csv (Dataset)")
        print("  - analysis_dashboard.png (EDA visualizations)")
        print("  - model_evaluation.png (Model performance)")


# Main execution
if __name__ == "__main__":
    system = StudentPerformanceSystem()
    system.run()
    
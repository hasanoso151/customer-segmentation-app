
import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# عنوان التطبيق
st.title("تصنيف عملاء مركز التسوق 🛍️🤖")

# وصف التطبيق
st.write("""
### أدخل بيانات العميل لتصنيفه إلى فئة VIP، عادي، أو غير نشط.
""")

# خطوة 1: تحميل البيانات
@st.cache_data  # تخزين البيانات مؤقتًا لتحسين الأداء
def load_data():
    # رابط البيانات المباشر
    url = "https://raw.githubusercontent.com/hasanoso151/costumer_project/refs/heads/main/Mall_Customers.csv"
    data = pd.read_csv(url)
    return data

data = load_data()

# خطوة 2: تحضير البيانات
def prepare_data(data):
    # إزالة الأعمدة غير الضرورية (CustomerID فقط)
    data_cleaned = data.drop(columns=['CustomerID'])
    
    # تحويل الجنس إلى أرقام
    label_encoder_gender = LabelEncoder()
    data_cleaned['Gender'] = label_encoder_gender.fit_transform(data_cleaned['Gender'])
    
    # إنشاء فئات العملاء بناءً على درجة الإنفاق
    data_cleaned['Category'] = pd.cut(
        data_cleaned['Spending Score (1-100)'],
        bins=[0, 30, 70, 100],
        labels=['غير نشط', 'عادي', 'VIP']
    )
    
    # تحويل الفئة (Category) إلى أرقام
    label_encoder_category = LabelEncoder()
    data_cleaned['Category'] = label_encoder_category.fit_transform(data_cleaned['Category'])
    
    # تقسيم البيانات إلى ميزات وهدف
    X = data_cleaned[['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']]  # جميع الخصائص
    y = data_cleaned['Category']
    
    # تقسيم البيانات إلى تدريب واختبار
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, label_encoder_category

X_train, X_test, y_train, y_test, label_encoder = prepare_data(data)

# خطوة 3: بناء النموذج
@st.cache_resource  # تخزين النموذج مؤقتًا لتحسين الأداء
def train_model(X_train, y_train):
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    return model

model = train_model(X_train, y_train)

# خطوة 4: واجهة المستخدم
gender = st.selectbox("الجنس", ["ذكر", "أنثى"])
age = st.number_input("العمر", min_value=0, max_value=100, value=30)
annual_income = st.number_input("الدخل السنوي (بالآلاف)", min_value=0, max_value=200, value=50)
spending_score = st.number_input("درجة الإنفاق (1-100)", min_value=0, max_value=100, value=50)

if st.button("تصنيف العميل"):
    # تحويل الجنس إلى رقم
    gender_encoded = 1 if gender == "أنثى" else 0
    
    # إعداد بيانات الإدخال
    input_data = pd.DataFrame({
        'Gender': [gender_encoded],
        'Age': [age],
        'Annual Income (k$)': [annual_income],
        'Spending Score (1-100)': [spending_score]
    })
    
    # التنبؤ
    prediction = model.predict(input_data)
    predicted_category = label_encoder.inverse_transform(prediction)
    
    # عرض النتيجة
    st.success(f"الفئة المتوقعة: {predicted_category[0]}")

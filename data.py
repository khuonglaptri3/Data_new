
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd  # Thư viện xử lý dữ liệu dạng bảng
from ucimlrepo import fetch_ucirepo

# Tải tập dữ liệu từ UCI Machine Learning Repository
data = fetch_ucirepo(id=350)
X = data.data.features

# Đổi tên cột để dễ hiểu hơn
X.rename(columns={
    'X2': 'Gender', 'X3': 'Education', 'X4': 'Marriage', 'X5': 'Age'
}, inplace=True)

# 1. Thống kê mô tả các biến nhân khẩu học
print("Thống kê mô tả các biến nhân khẩu học:")
print(X[['Gender', 'Education', 'Marriage', 'Age']].describe())

# 2. Trực quan hóa

# Giới tính
plt.figure(figsize=(12, 4))
plt.subplot(131)
sns.countplot(data=X, x='Gender', palette='pastel')
plt.title('Phân phối giới tính')
plt.xlabel('Giới tính (1 = Nam, 2 = Nữ)')
plt.ylabel('Số lượng')

# Trình độ học vấn
plt.subplot(132)
sns.countplot(data=X, x='Education', palette='pastel')
plt.title('Trình độ học vấn')
plt.xlabel('Trình độ học vấn (1 = Sau đại học, 2 = Đại học, 3 = Trung học, 4 = Khác)')
plt.ylabel('Số lượng')

# Tình trạng hôn nhânclscls
plt.subplot(133)
sns.countplot(data=X, x='Marriage', palette='pastel')
plt.title(' TTình trạng hôn nhân')
plt.xlabel('Tình trạng hôn nhân (1 = Đã kết hôn, 2 = Độc thân, 3 = Khác)')
plt.ylabel('Số lượng')
plt.tight_layout()
plt.show()

# Tuổi
plt.figure(figsize=(8, 5))
sns.histplot(data=X, x='Age', kde=True, color='skyblue', bins=30)
plt.title('Biểu đồ tuổi khách hàng')
plt.xlabel('Tuổi')
plt.ylabel('Số lượng')
plt.show()

# Biểu đồ tròn (Giới tính)
gender_counts = X['Gender'].value_counts()
plt.figure(figsize=(6, 6))
gender_counts.plot.pie(autopct='%1.1f%%', colors=['lightblue', 'lightpink'], labels=['Nam', 'Nữ'])
plt.title('Tỷ lệ giới tính')
plt.ylabel('')
plt.show()

# Biểu đồ đường (Tuổi)
age_distribution = X['Age'].value_counts().sort_index()
plt.figure(figsize=(10, 6))
plt.plot(age_distribution.index, age_distribution.values, marker='o', color='purple')
plt.title('Biểu đồ tuổi khách hàng')
plt.xlabel('Tuổi')
plt.ylabel('Số lượng')
plt.grid()
plt.show()
# Đổi tên cột để dễ hiểu hơn
X.rename(columns={
    'X6': 'Pay_Sept', 'X7': 'Pay_Aug', 'X8': 'Pay_Jul', 'X9': 'Pay_Jun', 'X10': 'Pay_May', 'X11': 'Pay_Apr',
    'X12': 'Bill_Sept', 'X13': 'Bill_Aug', 'X14': 'Bill_Jul', 'X15': 'Bill_Jun', 'X16': 'Bill_May', 'X17': 'Bill_Apr',
    'X18': 'PayAmt_Sept', 'X19': 'PayAmt_Aug', 'X20': 'PayAmt_Jul', 'X21': 'PayAmt_Jun', 'X22': 'PayAmt_May', 'X23': 'PayAmt_Apr'
}, inplace=True)

# 1. Thống kê mô tả
print("\nThống kê mô tả lịch sử thanh toán:")
print(X[['Pay_Sept', 'Pay_Aug', 'Pay_Jul', 'Pay_Jun', 'Pay_May', 'Pay_Apr']].describe())

print("\nThống kê mô tả số dư hóa đơn:")
print(X[['Bill_Sept', 'Bill_Aug', 'Bill_Jul', 'Bill_Jun', 'Bill_May', 'Bill_Apr']].describe())

print("\nThống kê mô tả số tiền đã thanh toán:")
print(X[['PayAmt_Sept', 'PayAmt_Aug', 'PayAmt_Jul', 'PayAmt_Jun', 'PayAmt_May', 'PayAmt_Apr']].describe())

# 2. Trực quan hóa

# Lịch sử thanh toán
plt.figure(figsize=(10, 6))
sns.boxplot(data=X[['Pay_Sept', 'Pay_Aug', 'Pay_Jul', 'Pay_Jun', 'Pay_May', 'Pay_Apr']], palette='pastel')
plt.title('Phân phối lịch sử thanh toán (PAY_X)')
plt.xlabel('Tháng')
plt.ylabel('Trạng thái thanh toán')
plt.show()

# Số dư hóa đơn
plt.figure(figsize=(12, 6))
sns.boxplot(data=X[['Bill_Sept', 'Bill_Aug', 'Bill_Jul', 'Bill_Jun', 'Bill_May', 'Bill_Apr']], palette='muted')
plt.title('Phân phối số dư hóa đơn (BILL_AMT_X)')
plt.xlabel('Tháng')
plt.ylabel('Số dư hóa đơn (NT$)')
plt.yscale('log')  # Hiển thị trên thang log để dễ quan sát
plt.show()

# Số tiền thanh toán
plt.figure(figsize=(12, 6))
sns.boxplot(data=X[['PayAmt_Sept', 'PayAmt_Aug', 'PayAmt_Jul', 'PayAmt_Jun', 'PayAmt_May', 'PayAmt_Apr']], palette='Set3')
plt.title('Phân phối số tiền đã thanh toán (PAY_AMT_X)')
plt.xlabel('Tháng')
plt.ylabel('Số tiền thanh toán (NT$)')
plt.yscale('log')  # Hiển thị trên thang log
plt.show()

# Biểu đồ đường: Số dư hóa đơn theo thời gian
avg_bills = X[['Bill_Sept', 'Bill_Aug', 'Bill_Jul', 'Bill_Jun', 'Bill_May', 'Bill_Apr']].mean()
plt.figure(figsize=(10, 6))
plt.plot(avg_bills.index, avg_bills.values, marker='o', color='blue')
plt.title('Số dư hóa đơn trung bình theo thời gian')
plt.xlabel('Tháng')
plt.ylabel('Số dư hóa đơn trung bình (NT$)')
plt.grid()
plt.show()

# Biểu đồ tròn: Trạng thái thanh toán tháng 9
pay_sept_counts = X['Pay_Sept'].value_counts()
plt.figure(figsize=(6, 6))
pay_sept_counts.plot.pie(autopct='%1.1f%%', colors=sns.color_palette('pastel'), labels=pay_sept_counts.index)
plt.title('Tỷ lệ trạng thái thanh toán tháng 9')
plt.ylabel('')
plt.show()
# Biểu đồ cột: Lịch sử thanh toán cho từng tháng
plt.figure(figsize=(10, 6))
pay_status = X[['Pay_Sept', 'Pay_Aug', 'Pay_Jul', 'Pay_Jun', 'Pay_May', 'Pay_Apr']].mean()
pay_status.plot(kind='bar', color='skyblue')
plt.title('Trung bình trạng thái thanh toán cho các tháng')
plt.xlabel('Tháng')
plt.ylabel('Trạng thái thanh toán trung bình')
plt.xticks(rotation=45)
plt.show()

# Biểu đồ cột: Số dư hóa đơn trung bình cho từng tháng
plt.figure(figsize=(10, 6))
bill_amounts = X[['Bill_Sept', 'Bill_Aug', 'Bill_Jul', 'Bill_Jun', 'Bill_May', 'Bill_Apr']].mean()
bill_amounts.plot(kind='bar', color='lightgreen')
plt.title('Số dư hóa đơn trung bình cho các tháng')
plt.xlabel('Tháng')
plt.ylabel('Số dư hóa đơn trung bình (NT$)')
plt.xticks(rotation=45)
plt.show()

# Biểu đồ cột: Số tiền đã thanh toán trung bình cho từng tháng
plt.figure(figsize=(10, 6))
payment_amounts = X[['PayAmt_Sept', 'PayAmt_Aug', 'PayAmt_Jul', 'PayAmt_Jun', 'PayAmt_May', 'PayAmt_Apr']].mean()
payment_amounts.plot(kind='bar', color='lightcoral')
plt.title('Số tiền thanh toán trung bình cho các tháng')
plt.xlabel('Tháng')
plt.ylabel('Số tiền thanh toán trung bình (NT$)')
plt.xticks(rotation=45)
plt.show()

# Biểu đồ đường: Sự thay đổi trung bình trạng thái thanh toán theo thời gian
plt.figure(figsize=(10, 6))
pay_status.plot(kind='line', marker='o', color='purple')
plt.title('Sự thay đổi trung bình trạng thái thanh toán theo thời gian')
plt.xlabel('Tháng')
plt.ylabel('Trạng thái thanh toán trung bình')
plt.grid()
plt.show()

# Biểu đồ đường: Sự thay đổi số dư hóa đơn trung bình theo thời gian
plt.figure(figsize=(10, 6))
bill_amounts.plot(kind='line', marker='o', color='blue')
plt.title('Sự thay đổi số dư hóa đơn trung bình theo thời gian')
plt.xlabel('Tháng')
plt.ylabel('Số dư hóa đơn trung bình (NT$)')
plt.grid()
plt.show()

# Biểu đồ đường: Sự thay đổi số tiền thanh toán trung bình theo thời gian
plt.figure(figsize=(10, 6))
payment_amounts.plot(kind='line', marker='o', color='orange')
plt.title('Sự thay đổi số tiền thanh toán trung bình theo thời gian')
plt.xlabel('Tháng')
plt.ylabel('Số tiền thanh toán trung bình (NT$)')
plt.grid()
plt.show()

# Biểu đồ tròn: Tỷ lệ trạng thái thanh toán tháng 9
pay_sept_counts = X['Pay_Sept'].value_counts()
plt.figure(figsize=(6, 6))
pay_sept_counts.plot.pie(autopct='%1.1f%%', colors=sns.color_palette('pastel'), labels=pay_sept_counts.index)
plt.title('Tỷ lệ trạng thái thanh toán tháng 9')
plt.ylabel('')
plt.show()
# 1. Phân tích mối quan hệ giữa Y và các biến phân loại (Giới tính, Trình độ học vấn, Tình trạng hôn nhân)
# Biểu đồ cột: Mối quan hệ giữa Y và Giới tính
plt.figure(figsize=(8, 6))
sns.countplot(x='Sex', hue='Y', data=X, palette='Set2')
plt.title('Mối quan hệ giữa Y và Giới tính')
plt.xlabel('Giới tính (1 = Nam, 2 = Nữ)')
plt.ylabel('Số lượng')
plt.show()

# Biểu đồ cột: Mối quan hệ giữa Y và Trình độ học vấn
plt.figure(figsize=(8, 6))
sns.countplot(x='Education Level', hue='Y', data=X, palette='Set2')
plt.title('Mối quan hệ giữa Y và Trình độ học vấn')
plt.xlabel('Trình độ học vấn')
plt.ylabel('Số lượng')
plt.show()

# Biểu đồ cột: Mối quan hệ giữa Y và Tình trạng hôn nhân
plt.figure(figsize=(8, 6))
sns.countplot(x='Marital Status', hue='Y', data=X, palette='Set2')
plt.title('Mối quan hệ giữa Y và Tình trạng hôn nhân')
plt.xlabel('Tình trạng hôn nhân (1 = Đã kết hôn, 2 = Độc thân, 3 = Khác)')
plt.ylabel('Số lượng')
plt.show()

# 2. Phân tích mối quan hệ giữa Y và các biến tài chính (Số dư hóa đơn, Số tiền thanh toán)
# Biểu đồ hộp: Mối quan hệ giữa Y và Số dư hóa đơn
plt.figure(figsize=(10, 6))
sns.boxplot(x='Y', y='Bill_Sept', data=X, palette='coolwarm')
plt.title('Mối quan hệ giữa Y và Số dư hóa đơn (Tháng 9)')
plt.xlabel('Biến mục tiêu (Y)')
plt.ylabel('Số dư hóa đơn tháng 9 (NT$)')
plt.show()

# Biểu đồ hộp: Mối quan hệ giữa Y và Số tiền thanh toán
plt.figure(figsize=(10, 6))
sns.boxplot(x='Y', y='PayAmt_Sept', data=X, palette='coolwarm')
plt.title('Mối quan hệ giữa Y và Số tiền thanh toán (Tháng 9)')
plt.xlabel('Biến mục tiêu (Y)')
plt.ylabel('Số tiền thanh toán tháng 9 (NT$)')
plt.show()

# 3. Phân tích mối quan hệ giữa Y và Lịch sử thanh toán
# Biểu đồ hộp: Mối quan hệ giữa Y và Lịch sử thanh toán (Pay_Sept)
plt.figure(figsize=(10, 6))
sns.boxplot(x='Y', y='Pay_Sept', data=X, palette='coolwarm')
plt.title('Mối quan hệ giữa Y và Lịch sử thanh toán (Tháng 9)')
plt.xlabel('Biến mục tiêu (Y)')
plt.ylabel('Trạng thái thanh toán tháng 9')
plt.show()

# Biểu đồ đường: Mối quan hệ giữa Y và Sự thay đổi số dư hóa đơn
avg_bill_by_Y = X.groupby('Y')[['Bill_Sept', 'Bill_Aug', 'Bill_Jul', 'Bill_Jun', 'Bill_May', 'Bill_Apr']].mean()
plt.figure(figsize=(10, 6))
avg_bill_by_Y.loc[0].plot(kind='line', marker='o', label='Y = 0', color='green')
avg_bill_by_Y.loc[1].plot(kind='line', marker='o', label='Y = 1', color='red')
plt.title('Sự thay đổi số dư hóa đơn trung bình theo thời gian theo Y')
plt.xlabel('Tháng')
plt.ylabel('Số dư hóa đơn trung bình (NT$)')
plt.legend()
plt.grid()
plt.show()

# Biểu đồ đường: Mối quan hệ giữa Y và Sự thay đổi số tiền thanh toán
avg_payment_by_Y = X.groupby('Y')[['PayAmt_Sept', 'PayAmt_Aug', 'PayAmt_Jul', 'PayAmt_Jun', 'PayAmt_May', 'PayAmt_Apr']].mean()
plt.figure(figsize=(10, 6))
avg_payment_by_Y.loc[0].plot(kind='line', marker='o', label='Y = 0', color='green')
avg_payment_by_Y.loc[1].plot(kind='line', marker='o', label='Y = 1', color='red')
plt.title('Sự thay đổi số tiền thanh toán trung bình theo thời gian theo Y')
plt.xlabel('Tháng')
plt.ylabel('Số tiền thanh toán trung bình (NT$)')
plt.legend()
plt.grid()
plt.show()

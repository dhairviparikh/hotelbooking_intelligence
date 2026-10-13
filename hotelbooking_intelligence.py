import pandas as pd

try:
    df=pd.read_csv('/content/hotelbooking_intelligence.csv')
except:
    print("Sample data use kar raha hai")
    data = {
        'Adult':[1,2,3,4,2,3,2,3,],
        'Childrens':[3,5,6,0,0,0,0,0],
        'Babies':[0,0,0,0,0,0,0,0]
    }
    df=pd.DataFrame(data)
df['Total_Guest']=df['Adult']+df['Childrens']+df['Babies']
df['Has_Children']=(df['Childrens']+df['Babies']>0).astype(int)

def make_label(row):
    if row['Total_Guest']==1:
        return "Single"
    elif row['Total_Guest']==2 and row['Has_Children']==0:
        return "Couples"
    else:
        return "Family"

df['Guest_Type']=df.apply(make_label,axis=1)
print(df.head(10))
print("\nCOUNT:")
print(df['Guest_Type'].value_counts())

#model tranning
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,classification_report
x=df[['Adult','Childrens','Babies','Total_Guest','Has_Children']]
y=df['Guest_Type']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
model=DecisionTreeClassifier(random_state=42)
model.fit(x_train,y_train)
pred=model.predict(x_test)
print(f"Accurancy:{accuracy_score(y_test,pred)*100:.2f}%")
print(classification_report(y_test,pred))



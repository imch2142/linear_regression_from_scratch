import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv('Ecommerce Customers.csv')
# plt.scatter(data['Time on Website'], data['Yearly Amount Spent'])
# plt.show()

def loss_function(m,b,n):
    total_error=0
    for i in range(len(n)):
        x=n.iloc[i]['Time on Website']
        y=n.iloc[i]['Yearly Amount Spent']
        total_error += (y - (m*x + b)) ** 2
    return total_error / len(n)


def gradient_descent(m,b,learning_rate,n):
    m_gradient=0
    b_gradient=0
    N=len(n)
    for i in range(len(n)):
        x=n.iloc[i]['Time on Website']
        y=n.iloc[i]['Yearly Amount Spent']
        m_gradient += (-2/N) * x * (y - (m*x + b))
        b_gradient += (-2/N) * (y - (m*x + b))
    new_m = m - (learning_rate * m_gradient)
    new_b = b - (learning_rate * b_gradient)

    return new_m, new_b


m=0
b=0
learning_rate=0.0001
iterations=30
for i in range(iterations):
    m,b=gradient_descent(m,b,learning_rate,data)
    if i % 50 == 0:
        print(f"Iteration {i}: Loss = {loss_function(m,b,data)}")

    print(m,b)
    plt.scatter(data['Time on Website'], data['Yearly Amount Spent'])
    plt.plot(data['Time on Website'], m*data['Time on Website'] + b, color='red')
    plt.show()
    plt.clf()
print(f"function : {m}*x+{b}")
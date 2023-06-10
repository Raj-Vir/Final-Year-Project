
# Importing the necessary library
from tensorflow.python.keras.models import load_model
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt



# Import the pretrained model
union_model = load_model("python_script/saved_models/UNIONBANK_model_1.h5",compile=False)
sbi_model = load_model("python_script/saved_models/SBIN_model.h5",compile=False)
bob_model = load_model("python_script/saved_models/BANKBARODA_model.h5",compile=False)
pnb_model = load_model("python_script/saved_models/PNB_model.h5",compile=False)
google_model = load_model("python_script/saved_models/GOOGLE_model_1.h5",compile=False)
apple_model = load_model("python_script/saved_models/APPLE_model.h5",compile=False)
msft_model = load_model("python_script/saved_models/MSFT_model.h5",compile=False)

# Importing the data for the models
union_data = pd.read_csv("Data/Stock_data/UNIONBANK_5Y.csv")
sbi_data = pd.read_csv("Data/Stock_data/SBIN_5Y.csv")
bob_data = pd.read_csv("Data/Stock_data/BANKBARODA_5Y.csv")
pnb_data = pd.read_csv("Data/Stock_data/PNB_5Y.csv")
google_data = pd.read_csv("Data/Stock_data/GOOG_5Y_1.csv")
apple_data = pd.read_csv("Data/Stock_data/APPLE_5Y.csv")
msft_data = pd.read_csv("Data/Stock_data/MSFT_5Y.csv")


# creating dictionary for the models
allmodels = {
            'Union Bank of India': union_model, 
            'State Bank of India': sbi_model,
            'Bank of Baroda': bob_model,
            'Punjab National Bank': pnb_model,
            'Google Corporation': google_model,
            'Apple Corporation': apple_model,
             'Microsoft Corporation': msft_model,
             }


# creating dictionary for the data
stocks = {
    'Union Bank of India': union_data, 
    'State Bank of India': sbi_data,
    'Bank of Baroda': bob_data,
    'Punjab National Bank': pnb_data,
    'Google Corporation': google_data,
    'Apple Corporation': apple_data,
    'Microsoft Corporation': msft_data,
    }

# Listing all the models for displaying
stocks_data = (
    'Union Bank of India', 
    'State Bank of India', 
    'Bank of Baroda',
    'Punjab National Bank',
    'Google Corporation', 
    'Apple Corporation',
    'Microsoft Corporation',
    )



# sidebar display
def choose_dataset(stocks, stocks_data, allmodels):
    st.sidebar.subheader('Select the Stock listed')
    stock = st.sidebar.selectbox( "", stocks_data, key='1' )
    check = st.sidebar.checkbox("Hide", value=True, key='0')
    exitPage = st.sidebar.checkbox("Exit", value=False, key='2')

    #st.sidebar.write(check)
    for itr in stocks_data:
        if stock==itr:
            main_df=stocks[itr]
            model=allmodels[itr]
    return main_df, check, stock, model, exitPage



   
# splitting the dataset
def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return np.array(dataX), np.array(dataY)


# Plotting the basic graph using data
def plot_predict(df, model, name):
    df = df.drop(["Open", "Low", "Adj Close", "Volume"], axis=1)
    df = df.dropna()
    Date = df["Date"]
    close = df["Close"]
    close = close.dropna()
    scaler = MinMaxScaler()
    tmp = scaler.fit(np.array(close).reshape(-1,1))
    new_df = scaler.fit_transform(np.array(close).reshape(-1,1))
    
    training_size=int(len(new_df)*0.67)
    test_size=len(new_df)-training_size
    train_data,test_data=new_df[:training_size],new_df[training_size:]
    Date_train, Date_test = Date[:training_size], Date[training_size:]
    
    n_steps = 30
    time_step=n_steps
    X_train, Y_train = create_dataset(train_data, time_step)
    X_test, Y_test = create_dataset(test_data, time_step)
    print('The Shape in plot predict before reshape:')
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
    X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)
    print('The Shape in plot predict after reshape:')
    print(X_train.shape, X_test.shape)
    
    
    
    
    train_predict=model.predict(X_train)
    test_predict=model.predict(X_test)
    print('train and test predict shape:')
    print(train_predict.shape, test_predict.shape)
    
   
    print(f'Train error - {mean_squared_error(train_predict, Y_train)}')
    print(f'Test error - {mean_squared_error(test_predict, Y_test)}')
    
    train_predict=scaler.inverse_transform(train_predict)
    test_predict=scaler.inverse_transform(test_predict)
    X_train=X_train.reshape(-1, 1)
    X_test=X_test.reshape(-1, 1)
    close_train=scaler.inverse_transform(train_data)
    close_test=scaler.inverse_transform(test_data)
    close_train = close_train.reshape(-1)
    close_test = close_test.reshape(-1)
    prediction = test_predict.reshape((-1))
    
    trace1 = go.Scatter(
        x = Date_train,
        y = close_train,
        mode = 'lines',
        name = 'Data'
    )
    trace2 = go.Scatter(
        x = Date_test[n_steps:],
        y = prediction,
        mode = 'lines',
        name = 'Prediction'
    )
    trace3 = go.Scatter(
        x = Date_test,
        y = close_test,
        mode='lines',
        name = 'Ground Truth'
    )
    layout = go.Layout(
        title = name,
        xaxis = {'title' : "Date"},
        yaxis = {'title' : "Close"}
    )
    fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)
    

    st.plotly_chart(fig)
    #fig.show()
    
    

# Plotting the forecasted graph by model
def plot_forecast_data(df, days, model, name):
    df = df.drop(["Open", "Low", "Adj Close", "Volume"], axis=1)
    df = df.dropna()
    Date = df["Date"]
    close = df["Close"]
    close = close.dropna()
    scaler = MinMaxScaler()
    tmp = scaler.fit(np.array(close).reshape(-1,1))
    new_df = scaler.transform(np.array(close).reshape(-1,1))
    
    
    
    test_data = close
    test_data = scaler.fit_transform(np.array(close).reshape(-1,1))
    test_data = test_data.reshape((-1))
    
    def predict(num_prediction, model):
        prediction_list = test_data[-n_steps:]
        
        for _ in range(num_prediction):
            x = prediction_list[-n_steps:]
            x = x.reshape((1, n_steps, 1))
            out = model.predict(x)
            prediction_list = np.append(prediction_list, out)
        prediction_list = prediction_list[n_steps-1:]
            
        return prediction_list
        
    def predict_dates(num_prediction):
        last_date = df['Date'].values[-1]
        prediction_dates = pd.date_range(last_date, periods=num_prediction+1).tolist()
        return prediction_dates
    
    num_prediction =days
    forecast = predict(num_prediction, model)
    forecast_dates = predict_dates(num_prediction)
    forecast = forecast.reshape(1, -1)
    forecast = scaler.inverse_transform(forecast)
    forecast
    test_data = test_data.reshape(1, -1)
    test_data = scaler.inverse_transform(test_data)
    test_data = test_data.reshape(-1)
    forecast = forecast.reshape(-1)
    res = dict(zip(forecast_dates, forecast))
    date = df["Date"]
    trace1 = go.Scatter(
        x = date,
        y = test_data,
        mode = 'lines',
        name = 'Data'
    )
    trace2 = go.Scatter(
        x = forecast_dates,
        y = forecast,
        mode = 'lines',
        name = 'Prediction'
    )
    layout = go.Layout(
    title = name,
    xaxis = {'title' : "Date"},
    yaxis = {'title' : "Close"}
    )
    
    fig = go.Figure(data=[trace1, trace2], layout=layout)
    st.plotly_chart(fig)
    #fig.show()
    choose_date = st.selectbox("Date", forecast_dates)
    for itr in res:
        if choose_date==itr:
            res_price=res[itr]
    st.write(f"On {choose_date} the stock price will be: {res_price}")

    
    
    
    



# Plotting the graph with opening and closing price comparison  
def plot_raw_data(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.layout.update( xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    	
# Landing UI
def landing_ui():
    st.header("Welcome to Stock Price Prediction : A smart way to invest your money")
    st.write("")
    st.write("")
    st.write("Welcome to this site")
    st.write("As the model is trained with data having time steps of 30 days so it will give its best results for a forecast till 30days ")
    st.write("")
    st.write("To see the data representation please uncheck the hide button in the sidebar")
    st.write("")
    st.write("Share market investments are subject to market risks, read all scheme related documents carefully. The NAVs of the schemes may go up or down depending upon the factors and forces affecting the securities market including the fluctuations in the interest rates. The past performance of the stocks is not necessarily indicative of future performance of the schemes.")

def exiting_ui():
    st.header("BYE")
    st.write("")
    st.write("")
    st.write("Thank You For visiting")
    
# Genetic Algorithm Implementation
n_steps = 30

class neuralnetwork:
        def __init__(self, id_, hidden_size = 128):
            self.W1 = np.random.randn(window_size, hidden_size) / np.sqrt(window_size)
            self.W2 = np.random.randn(hidden_size, 3) / np.sqrt(hidden_size)
            self.fitness = 0
            self.id = id_   

def relu(X):
    return np.maximum(X, 0)

def softmax(X):
    e_x = np.exp(X - np.max(X, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def feed_forward(X, nets):
    a1 = np.dot(X, nets.W1)
    z1 = relu(a1)
    a2 = np.dot(z1, nets.W2)
    return softmax(a2)
class NeuroEvolution:
    def __init__(self, population_size, mutation_rate, model_generator,state_size, window_size, trend, skip, initial_money):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.model_generator = model_generator
        self.state_size = state_size
        self.window_size = window_size
        self.half_window = window_size // 2
        self.trend = trend
        self.skip = skip
        self.initial_money = initial_money
        
    def _initialize_population(self):
        self.population = []
        for i in range(self.population_size):
            self.population.append(self.model_generator(i))

    def mutate(self, individual, scale=1.0):
        mutation_mask = np.random.binomial(1, p=self.mutation_rate, size=individual.W1.shape)
        individual.W1 += np.random.normal(loc=0, scale=scale, size=individual.W1.shape) * mutation_mask
        mutation_mask = np.random.binomial(1, p=self.mutation_rate, size=individual.W2.shape)
        individual.W2 += np.random.normal(loc=0, scale=scale, size=individual.W2.shape) * mutation_mask
        return individual

    def inherit_weights(self, parent, child):
        child.W1 = parent.W1.copy()
        child.W2 = parent.W2.copy()
        return child

    def crossover(self, parent1, parent2):
        child1 = self.model_generator((parent1.id+1)*10)
        child1 = self.inherit_weights(parent1, child1)
        child2 = self.model_generator((parent2.id+1)*10)
        child2 = self.inherit_weights(parent2, child2)
        # first W
        n_neurons = child1.W1.shape[1]
        cutoff = np.random.randint(0, n_neurons)
        child1.W1[:, cutoff:] = parent2.W1[:, cutoff:].copy()
        child2.W1[:, cutoff:] = parent1.W1[:, cutoff:].copy()
        # second W
        n_neurons = child1.W2.shape[1]
        cutoff = np.random.randint(0, n_neurons)
        child1.W2[:, cutoff:] = parent2.W2[:, cutoff:].copy()
        child2.W2[:, cutoff:] = parent1.W2[:, cutoff:].copy()
        return child1, child2

    def get_state(self, t):
        window_size = self.window_size + 1
        d = t - window_size + 1
        block = self.trend[d : t + 1] if d >= 0 else -d * [self.trend[0]] + self.trend[0 : t + 1]
        res = []
        for i in range(window_size - 1):
            res.append(block[i + 1] - block[i])
        return np.array([res])

    def act(self, p, state):
        logits = feed_forward(state, p)
        return np.argmax(logits, 1)[0]

    def buy(self, individual, trends):
        initial_money = self.initial_money
        starting_money = initial_money
        state = self.get_state(0)
        inventory = []
        states_sell = []
        states_buy = []
        
        for t in range(0, len(trends) - 1, self.skip):
            action = self.act(individual, state)
            next_state = self.get_state(t + 1)
            
            if action == 1 and starting_money >= trends[t]:
                inventory.append(trends[t])
                initial_money -= trends[t]
                states_buy.append(t)
                # print('day %d: buy 1 unit at price %f, total balance %f'% (t, trends[t], initial_money))
            
            elif action == 2 and len(inventory):
                bought_price = inventory.pop(0)
                initial_money += trends[t]
                states_sell.append(t)
                try:
                    invest = ((trends[t] - bought_price) / bought_price) * 100
                except:
                    invest = 0
                # print('day %d, sell 1 unit at price %f, investment %f %%, total balance %f,'% (t, trends[t], invest, initial_money))
            state = next_state
        
        invest = ((initial_money - starting_money) / starting_money) * 100
        total_gains = initial_money - starting_money
        return states_buy, states_sell, total_gains, invest

    def calculate_fitness(self):
        for i in range(self.population_size):
            initial_money = self.initial_money
            starting_money = initial_money
            state = self.get_state(0)
            inventory = []
            
            for t in range(0, len(self.trend) - 1, self.skip):
                action = self.act(self.population[i], state)
                next_state = self.get_state(t + 1)
            
                if action == 1 and starting_money >= self.trend[t]:
                    inventory.append(self.trend[t])
                    starting_money -= self.trend[t]

                elif action == 2 and len(inventory):
                    bought_price = inventory.pop(0)
                    starting_money += self.trend[t]

                state = next_state
            invest = ((starting_money - initial_money) / initial_money) * 100
            self.population[i].fitness = invest
        

    def evolve(self, generations=20, checkpoint= 5):
        self._initialize_population()
        n_winners = int(self.population_size * 0.4)
        n_parents = self.population_size - n_winners
        for epoch in range(generations):
            self.calculate_fitness()
            fitnesses = [i.fitness for i in self.population]
            sort_fitness = np.argsort(fitnesses)[::-1]
            self.population = [self.population[i] for i in sort_fitness]
            fittest_individual = self.population[0]
            if (epoch+1) % checkpoint == 0:
                print('epoch %d, fittest individual %d with accuracy %f'%(epoch+1, sort_fitness[0], 
                                                                        fittest_individual.fitness))
            next_population = [self.population[i] for i in range(n_winners)]
            total_fitness = np.sum([np.abs(i.fitness) for i in self.population])
            parent_probabilities = [np.abs(i.fitness / total_fitness) for i in self.population]
            parents = np.random.choice(self.population, size=n_parents, p=parent_probabilities, replace=False)
            for i in np.arange(0, len(parents), 2):
                child1, child2 = self.crossover(parents[i], parents[i+1])
                next_population += [self.mutate(child1), self.mutate(child2)]
            self.population = next_population
        return fittest_individual
    
def plot_graph():
    fig = plt.figure(figsize=(15,5))
    plt.plot(close, color='r', lw=2.)
    plt.plot(close, '^', markersize=10, color='m', label = 'buying signal', markevery = states_buy)
    plt.plot(close, 'v', markersize=10, color='k', label = 'selling signal', markevery = states_sell)
    plt.title('total gains %f, total investment %f%%'%(total_gains, invest))
    plt.legend()
    st.pyplot(fig)


if __name__ == "__main__":
    
    st.sidebar.header("Stock Market Predictor")
    st.sidebar.markdown("---")
    temp, check, name, model,exitPage =choose_dataset(stocks, stocks_data, allmodels)
    #about_section()
    #print(temp)
    if not check and (not exitPage):
        st.header(f"Analyzing {name}'s stock data")
        st.subheader("Raw Data")
        st.write(temp)
        
        
        st.subheader("Raw Data - Visualized")
        plot_raw_data(temp)
        st.subheader("Predicted data")
        plot_predict(temp, model, name)
        st.sidebar.subheader("Forecasted Data")
        forecast_check = st.sidebar.checkbox("See the results", value=False)
        
        if forecast_check:
            forecast = st.slider("Days to forecast",min_value=30,max_value=100,step=5)
            st.subheader("Forecasted data")
            
            plot_forecast_data(temp, forecast, model, name)

            st.subheader("Forecasted Graph with entry and exit")
            initial_money = 50000
            window_size = 30
            df = temp
            df = df.head(100)
            df = df.drop("Date",axis=1)
            close = df.Close.values.tolist()
            skip = 1
            population_size = 100
            generations = 100
            mutation_rate = 0.1
            neural_evolve = NeuroEvolution(population_size, mutation_rate, neuralnetwork,window_size, window_size, close, skip, initial_money)
            fittest_nets = neural_evolve.evolve(50)
            states_buy, states_sell, total_gains, invest = neural_evolve.buy(fittest_nets,close)
            plot_graph()
    elif (check) and (not exitPage) :
        landing_ui()
    else:
        exiting_ui()

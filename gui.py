import ipywidgets as widgets
from IPython.display import display
import seaborn as sns


# Function to make predictions based on user input
def predict_price(open_price, high_price, low_price, volume, year, month, day):
    """
    Do not use-still under construction
    :param open_price:
    :param high_price:
    :param low_price:
    :param volume:
    :param year:
    :param month:
    :param day:
    :return:
    """
    input_data = np.array([[open_price, high_price, low_price, volume, year, month, day]])
    prediction = model.predict(input_data)
    print(f'Predicted Stock Price: {prediction[0]:.2f}')

# Creating the interactive widgets
open_price_slider = widgets.FloatSlider(value=150, min=100, max=200, step=1, description='Open Price:')
high_price_slider = widgets.FloatSlider(value=155, min=105, max=205, step=1, description='High Price:')
low_price_slider = widgets.FloatSlider(value=145, min=95, max=195, step=1, description='Low Price:')
volume_slider = widgets.FloatSlider(value=50000000, min=10000000, max=100000000, step=1000000, description='Volume:')
year_slider = widgets.IntSlider(value=2020, min=2015, max=2025, step=1, description='Year:')
month_slider = widgets.IntSlider(value=6, min=1, max=12, step=1, description='Month:')
day_slider = widgets.IntSlider(value=15, min=1, max=31, step=1, description='Day:')

# Displaying the widgets and connecting them to the prediction function
interactive_plot = widgets.interactive(predict_price,
                                       open_price=open_price_slider,
                                       high_price=high_price_slider,
                                       low_price=low_price_slider,
                                       volume=volume_slider,
                                       year=year_slider,
                                       month=month_slider,
                                       day=day_slider)
display(interactive_plot)

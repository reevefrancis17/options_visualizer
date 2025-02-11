from python.data.options_data import OptionsData
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, CheckButtons
import numpy as np

class OptionsVisualizer:
    def __init__(self):
        # Initialize options data
        self.options = OptionsData()
        self.options.fetch_data('AAPL')
        self.dates = self.options.get_available_dates()
        self.current_date_idx = 0
        
        # Display flags
        self.show_calls = True
        self.show_puts = True
        
        # Available fields for plotting
        self.fields = {
            'Mid Price': {'call': 'call_mid', 'put': 'put_mid'},
            'Bid': {'call': 'call_bid', 'put': 'put_bid'},
            'Ask': {'call': 'call_ask', 'put': 'put_ask'},
            'Last': {'call': 'call_price', 'put': 'put_price'},
            'Volume': {'call': 'call_volume', 'put': 'put_volume'},
            'Open Interest': {'call': 'call_oi', 'put': 'put_oi'},
            'Intrinsic Value': {'call': 'intrinsic_value', 'put': 'put_intrinsic_value'},
            'Extrinsic Value': {'call': 'extrinsic_value', 'put': 'put_extrinsic_value'}
        }
        self.current_field = 'Mid Price'
        
        # Create the plot
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(bottom=0.2, left=0.3)
        
        # Add navigation buttons
        self.ax_prev = plt.axes([0.7, 0.05, 0.1, 0.075])
        self.ax_next = plt.axes([0.81, 0.05, 0.1, 0.075])
        self.btn_prev = Button(self.ax_prev, 'Previous')
        self.btn_next = Button(self.ax_next, 'Next')
        
        # Add radio buttons for field selection
        self.ax_radio = plt.axes([0.05, 0.2, 0.2, 0.6])
        self.radio = RadioButtons(self.ax_radio, list(self.fields.keys()))
        
        # Add toggle buttons for calls/puts
        self.ax_toggle = plt.axes([0.05, 0.05, 0.2, 0.1])
        self.toggle = CheckButtons(self.ax_toggle, ['Calls', 'Puts'], [True, True])
        
        # Connect events
        self.btn_prev.on_clicked(self.prev_date)
        self.btn_next.on_clicked(self.next_date)
        self.radio.on_clicked(self.field_changed)
        self.toggle.on_clicked(self.toggle_clicked)
        
        self.update_plot()
        
    def update_plot(self):
        self.ax.clear()
        date = self.dates[self.current_date_idx]
        
        # Interpolate zeros before getting chain
        self.options._interpolate_zeros()
        chain = self.options.get_chain_for_date(date)
        
        # Extract strikes
        strikes = [opt['strike'] for opt in chain['options']]
        
        # Plot calls if enabled
        if self.show_calls:
            call_values = [opt[self.fields[self.current_field]['call']] for opt in chain['options']]
            self.ax.plot(strikes, call_values, 'b.-', label=f'Calls {self.current_field}')
        
        # Plot puts if enabled
        if self.show_puts:
            put_values = [opt[self.fields[self.current_field]['put']] for opt in chain['options']]
            self.ax.plot(strikes, put_values, 'r.-', label=f'Puts {self.current_field}')
        
        self.ax.set_xlabel('Strike Price')
        self.ax.set_ylabel(f'{self.current_field} ($)')
        self.ax.set_title(f'AAPL Options - {self.current_field} - Expiry: {date}')
        self.ax.grid(True)
        self.ax.legend()
        
        plt.draw()
    
    def toggle_clicked(self, label):
        if label == 'Calls':
            self.show_calls = not self.show_calls
        else:  # Puts
            self.show_puts = not self.show_puts
        self.update_plot()
    
    def field_changed(self, label):
        self.current_field = label
        self.update_plot()
    
    def prev_date(self, event):
        self.current_date_idx = max(0, self.current_date_idx - 1)
        self.update_plot()
    
    def next_date(self, event):
        self.current_date_idx = min(len(self.dates) - 1, self.current_date_idx + 1)
        self.update_plot()

if __name__ == '__main__':
    visualizer = OptionsVisualizer()
    plt.show() 
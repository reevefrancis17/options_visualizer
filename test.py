from python.data.options_data import OptionsData
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons
import numpy as np

class OptionsVisualizer:
    def __init__(self):
        # Initialize options data
        self.options = OptionsData()
        self.options.fetch_data('AAPL')
        self.dates = self.options.get_available_dates()
        self.current_date_idx = 0
        
        # Available fields for plotting
        self.fields = {
            'Call Mid Price': 'call_mid',
            'Call Bid': 'call_bid',
            'Call Ask': 'call_ask',
            'Call Last': 'call_price',
            'Intrinsic Value': 'intrinsic_value',
            'Extrinsic Value': 'extrinsic_value',
            'Volume': 'call_volume',
            'Open Interest': 'call_oi'
        }
        self.current_field = 'call_mid'
        
        # Create the plot
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(bottom=0.2, left=0.3)  # Make room for buttons and radio
        
        # Add buttons
        self.ax_prev = plt.axes([0.7, 0.05, 0.1, 0.075])
        self.ax_next = plt.axes([0.81, 0.05, 0.1, 0.075])
        self.btn_prev = Button(self.ax_prev, 'Previous')
        self.btn_next = Button(self.ax_next, 'Next')
        
        # Add radio buttons for field selection
        self.ax_radio = plt.axes([0.05, 0.2, 0.2, 0.6])
        self.radio = RadioButtons(self.ax_radio, list(self.fields.keys()))
        
        # Connect events
        self.btn_prev.on_clicked(self.prev_date)
        self.btn_next.on_clicked(self.next_date)
        self.radio.on_clicked(self.field_changed)
        
        self.update_plot()
        
    def update_plot(self):
        self.ax.clear()
        date = self.dates[self.current_date_idx]
        
        # Interpolate zeros before getting chain
        self.options._interpolate_zeros()
        chain = self.options.get_chain_for_date(date)
        
        # Extract strikes and selected field values
        strikes = [opt['strike'] for opt in chain['options']]
        values = [opt[self.current_field] for opt in chain['options']]
        
        # Plot
        self.ax.plot(strikes, values, 'b.-', label=self.get_field_label())
        self.ax.set_xlabel('Strike Price')
        self.ax.set_ylabel(f'{self.get_field_label()} ($)')
        self.ax.set_title(f'AAPL Options - {self.get_field_label()} - Expiry: {date}')
        self.ax.grid(True)
        self.ax.legend()
        
        plt.draw()
    
    def get_field_label(self):
        """Get display label for current field"""
        return next(k for k, v in self.fields.items() if v == self.current_field)
    
    def field_changed(self, label):
        """Handle radio button selection"""
        self.current_field = self.fields[label]
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
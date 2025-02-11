from python.data.options_data import OptionsData
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, CheckButtons, TextBox
import numpy as np

class OptionsVisualizer:
    def __init__(self):
        # Initialize options data
        self.options = OptionsData()
        self.symbol = 'AAPL'
        self.fetch_data()
        
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
        self.show_calls = True
        self.show_puts = True
        
        # Create the plot with adjusted spacing for ticker box
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(bottom=0.2, left=0.3, top=0.85)  # Added top margin
        
        # Add ticker input box at top right
        self.ax_ticker = plt.axes([0.7, 0.92, 0.2, 0.03])  # [left, bottom, width, height]
        self.ticker_box = TextBox(self.ax_ticker, 'Symbol: ', initial=self.symbol)
        
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
        
        # Initialize cursor lines as None
        self.v_line = None
        self.call_line = None
        self.put_line = None
        
        # Connect events
        self.btn_prev.on_clicked(self.prev_date)
        self.btn_next.on_clicked(self.next_date)
        self.radio.on_clicked(self.field_changed)
        self.toggle.on_clicked(self.toggle_clicked)
        self.ticker_box.on_submit(self.ticker_changed)
        
        # Connect mouse motion event
        self.cursor_id = self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        
        self.update_plot()
    
    def fetch_data(self):
        """Fetch data for current symbol"""
        try:
            self.options.fetch_data(self.symbol)
            self.dates = self.options.get_available_dates()
            self.current_date_idx = 0
        except Exception as e:
            print(f"Error fetching data for {self.symbol}: {str(e)}")
            # Revert to previous symbol if fetch fails
            if hasattr(self, 'dates'):
                self.symbol = self.options.symbol
            else:
                self.symbol = 'AAPL'
                self.options.fetch_data(self.symbol)
                self.dates = self.options.get_available_dates()
                self.current_date_idx = 0
    
    def ticker_changed(self, text):
        """Handle ticker symbol change"""
        new_symbol = text.upper().strip()
        if new_symbol != self.symbol:
            self.symbol = new_symbol
            self.fetch_data()
            self.update_plot()
    
    def update_plot(self):
        self.ax.clear()
        date = self.dates[self.current_date_idx]
        
        # Interpolate zeros before getting chain
        self.options._interpolate_zeros()
        self.chain = self.options.get_chain_for_date(date)
        
        # Extract strikes
        strikes = [opt['strike'] for opt in self.chain['options']]
        
        # Plot calls if enabled
        if self.show_calls:
            call_values = [opt[self.fields[self.current_field]['call']] for opt in self.chain['options']]
            self.call_plot = self.ax.plot(strikes, call_values, 'b.-', label=f'Calls {self.current_field}')[0]
        
        # Plot puts if enabled
        if self.show_puts:
            put_values = [opt[self.fields[self.current_field]['put']] for opt in self.chain['options']]
            self.put_plot = self.ax.plot(strikes, put_values, 'r.-', label=f'Puts {self.current_field}')[0]
        
        # Add spot price line
        spot = self.options.spot_price
        self.spot_line = self.ax.axvline(x=spot, color='g', linestyle='--')
        
        self.ax.set_xlabel('Strike Price')
        self.ax.set_ylabel(f'{self.current_field} ($)')
        self.ax.set_title(f'{self.symbol} Options - {self.current_field} - Expiry: {date}')
        self.ax.grid(True)
        
        # Initialize legend with basic elements
        self.ax.legend()
        
        # Reset cursor lines
        self.v_line = None
        self.call_line = None
        self.put_line = None
        
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
    
    def on_mouse_move(self, event):
        """Update crosshair and value indicators for calls and puts"""
        if event.inaxes != self.ax:
            return
            
        # Remove old lines safely
        for line in [self.v_line, self.call_line, self.put_line]:
            if line:
                line.remove()
        
        # Get nearest strike price
        strikes = [opt['strike'] for opt in self.chain['options']]
        strike_idx = min(range(len(strikes)), 
                        key=lambda i: abs(strikes[i] - event.xdata))
        strike = strikes[strike_idx]
        
        # Get call and put values for this strike
        opt_data = self.chain['options'][strike_idx]
        call_value = opt_data[self.fields[self.current_field]['call']]
        put_value = opt_data[self.fields[self.current_field]['put']]
        
        # Add vertical line at strike and horizontal lines at values
        self.v_line = self.ax.axvline(x=strike, color='gray', linestyle=':')
        if self.show_calls:
            self.call_line = self.ax.axhline(y=call_value, color='blue', linestyle=':')
        if self.show_puts:
            self.put_line = self.ax.axhline(y=put_value, color='red', linestyle=':')
        
        # Update legend with cursor values
        legend_elements = []
        legend_labels = []
        
        # Add spot price to legend
        spot = self.options.spot_price
        legend_elements.append(self.spot_line)
        legend_labels.append(f'Spot: {spot:.2f}')
        
        # Add main plot lines to legend
        if self.show_calls:
            legend_elements.append(self.call_plot)
            legend_labels.append(f'Calls {self.current_field}')
        if self.show_puts:
            legend_elements.append(self.put_plot)
            legend_labels.append(f'Puts {self.current_field}')
        
        # Add cursor values to legend
        legend_labels.append(f'Strike: {strike:.2f}')
        if self.show_calls:
            legend_labels.append(f'Call Value: {call_value:.2f}')
        if self.show_puts:
            legend_labels.append(f'Put Value: {put_value:.2f}')
        
        # Update legend
        self.ax.legend(legend_elements + [plt.Line2D([0], [0], color='none')]*3,
                      legend_labels,
                      loc='upper left')
        
        self.fig.canvas.draw_idle()

if __name__ == '__main__':
    visualizer = OptionsVisualizer()
    plt.show() 
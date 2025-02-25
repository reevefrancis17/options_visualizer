# offline_app.py
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.ticker import FuncFormatter
import pandas as pd
import numpy as np
import logging
import re
import time
import os
from yahoo_finance import YahooFinanceAPI
from options_data import OptionsDataProcessor
import traceback

# Clear the log file at startup
log_dir = 'debug'
log_file = os.path.join(log_dir, 'error_log.txt')

# Create the debug directory if it doesn't exist
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Clear the log file by opening it in write mode
with open(log_file, 'w') as f:
    f.write(f"=== New session started at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")

# Configure logging to write to the specified file
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
logger.info("Starting new session of Options Visualizer")

class OptionsVisualizerApp(tk.Tk):
    """
    GUI application titled "Options Visualizer" for displaying options data.
    Features a search bar, navigation buttons for expiry dates, and line plot of options data.
    """
    def __init__(self):
        super().__init__()
        self.title("Options Visualizer")
        self.geometry("1200x800")  # Increased size for better visibility
        self.minsize(1000, 600)    # Set minimum window size
        
        logger.info("Initializing Options Visualizer App")
        self.api = YahooFinanceAPI()
        self.data_processor = None
        self.expiry_dates = []
        self.current_expiry_index = 0
        self.current_ticker = ""
        self.last_update_time = 0
        self.data_loaded = False  # Track if data has been loaded
        
        self.create_widgets()
        
        # Schedule loading of default ticker after GUI is ready
        self.after(100, self.load_default_ticker)
        
    def create_widgets(self):
        # Top frame for ticker input and navigation
        top_frame = tk.Frame(self)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        # Left side - Ticker input
        ticker_frame = tk.Frame(top_frame)
        ticker_frame.pack(side=tk.LEFT)
        
        tk.Label(ticker_frame, text="Ticker:").pack(side=tk.LEFT)
        self.ticker_entry = tk.Entry(ticker_frame, width=15)
        self.ticker_entry.pack(side=tk.LEFT, padx=5)
        self.search_button = tk.Button(ticker_frame, text="Search", command=self.search_ticker)
        self.search_button.pack(side=tk.LEFT)
        
        # Status label for last update time
        self.status_label = tk.Label(ticker_frame, text="Not updated yet", font=("Arial", 8))
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        # Right side - Expiry navigation
        nav_frame = tk.Frame(top_frame)
        nav_frame.pack(side=tk.RIGHT, padx=20)
        
        self.prev_button = tk.Button(nav_frame, text="◀", command=self.prev_expiry, state=tk.DISABLED)
        self.prev_button.pack(side=tk.LEFT, padx=5)
        
        self.expiry_label = tk.Label(nav_frame, text="No data", width=12)
        self.expiry_label.pack(side=tk.LEFT, padx=5)
        
        self.next_button = tk.Button(nav_frame, text="▶", command=self.next_expiry, state=tk.DISABLED)
        self.next_button.pack(side=tk.LEFT, padx=5)
        
        # Create main container for plot
        main_container = tk.Frame(self)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Plot options
        self.plot_var = tk.StringVar(value="Spot")
        plot_options_frame = ttk.LabelFrame(main_container, text="Plot Type", padding=5)
        plot_options_frame.pack(fill=tk.X, pady=5)
        
        self.plot_options = [
            ("Spot", "Spot"),
            ("Last Price", "Last Price"),
            ("Bid", "Bid"),
            ("Ask", "Ask"),
            ("Volume", "Volume"),
            ("Intrinsic Value", "Intrinsic Value"),
            ("Extrinsic Value", "Extrinsic Value"),
        ]
        
        for text, value in self.plot_options:
            ttk.Radiobutton(plot_options_frame, text=text, value=value, 
                          variable=self.plot_var, command=self.update_plot).pack(side=tk.LEFT, padx=5)
        
        # Figure for matplotlib
        self.figure = plt.Figure(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.figure, main_container)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def prev_expiry(self):
        if self.current_expiry_index > 0:
            self.current_expiry_index -= 1
            self.update_expiry_display()
            self.update_plot()
    
    def next_expiry(self):
        if self.current_expiry_index < len(self.expiry_dates) - 1:
            self.current_expiry_index += 1
            self.update_expiry_display()
            self.update_plot()
    
    def update_expiry_display(self):
        if self.expiry_dates:
            current_date = self.expiry_dates[self.current_expiry_index]
            self.expiry_label.config(text=current_date.strftime('%Y-%m-%d'))
            self.prev_button.config(state=tk.NORMAL if self.current_expiry_index > 0 else tk.DISABLED)
            self.next_button.config(state=tk.NORMAL if self.current_expiry_index < len(self.expiry_dates) - 1 else tk.DISABLED)
        else:
            self.expiry_label.config(text="No data")
            self.prev_button.config(state=tk.DISABLED)
            self.next_button.config(state=tk.DISABLED)
    
    def load_default_ticker(self):
        """Load the default ticker (SPY) on startup"""
        logger.info("Loading default ticker: SPY")
        self.ticker_entry.insert(0, "SPY")
        
        # Show loading message immediately
        self.show_loading_message("Loading SPY options data...")
        
        # Use after to give the GUI time to render before loading data
        self.after(200, self._load_default_ticker_data)
    
    def _load_default_ticker_data(self):
        """Actually load the data for the default ticker"""
        ticker = "SPY"
        try:
            # Update status to show loading
            self.status_label.config(text="Loading data...")
            
            # Fetch data from Yahoo Finance with progressive loading
            logger.info(f"Fetching options data for {ticker}")
            options_data, current_price = self.api.get_options_data(ticker, 
                                                                   self.update_with_partial_data,
                                                                   max_dates=8)  # Limit to 8 expiration dates for faster loading
            
            if options_data is None or current_price is None:
                logger.error(f"Failed to fetch data for {ticker}")
                self.status_label.config(text="Failed to load default data")
                self.show_loading_message("Failed to load default data. Please try searching manually.")
                return
                
            # Final update with complete data
            self.update_with_complete_data(ticker, options_data, current_price)
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error loading default ticker: {error_msg}")
            self.status_label.config(text=f"Error: {error_msg[:20]}...")
            self.show_loading_message(f"Error loading default data: {error_msg[:50]}...\nPlease try searching manually.")
    
    def update_with_partial_data(self, partial_data, current_price, processed_dates, total_dates):
        """Update the UI with partial data as it's being loaded"""
        if not current_price:
            # Still waiting for price data
            self.show_loading_message(f"Fetching current price for {self.ticker_entry.get()}...")
            return
            
        if not partial_data:
            # Got price but no option data yet
            self.show_loading_message(f"Got price (${current_price:.2f}) for {self.ticker_entry.get()}\nFetching options data...")
            return
            
        # Update loading message to show progress
        progress_pct = int((processed_dates / total_dates) * 100)
        self.show_loading_message(f"Loading {self.ticker_entry.get()} options data...\n"
                                 f"Current price: ${current_price:.2f}\n"
                                 f"{progress_pct}% complete ({processed_dates}/{total_dates} expiry dates)")
        
        # If we have at least one expiration date with data, start processing and displaying it
        if processed_dates >= 1:
            try:
                # Process the partial data
                temp_processor = OptionsDataProcessor(partial_data, current_price)
                
                # Get expiration dates from partial data
                expiry_dates = temp_processor.get_expirations()
                if expiry_dates:
                    # Store the processor and dates temporarily
                    self.temp_data_processor = temp_processor
                    self.temp_expiry_dates = expiry_dates
                    
                    # Update the UI with the first available expiration date
                    self.after(10, self.update_with_temp_data)
            except Exception as e:
                logger.error(f"Error processing partial data: {str(e)}")
                # Continue loading - don't interrupt the process for partial data errors
    
    def update_with_temp_data(self):
        """Update the UI with temporary data while loading continues"""
        if not hasattr(self, 'temp_data_processor') or not hasattr(self, 'temp_expiry_dates'):
            return
            
        try:
            # Store the temporary data as the current data
            self.data_processor = self.temp_data_processor
            self.expiry_dates = self.temp_expiry_dates
            
            # Default to first expiry date (lowest DTE)
            self.current_expiry_index = 0
            
            # Update UI
            self.update_expiry_display()
            self.update_plot()
            
            # Add a "Loading..." indicator to the title
            if hasattr(self, 'figure') and hasattr(self.figure, 'axes') and self.figure.axes:
                ax = self.figure.axes[0]
                current_title = ax.get_title()
                ax.set_title(f"{current_title} (Loading more data...)")
                self.canvas.draw_idle()  # Use draw_idle for better performance
        except Exception as e:
            logger.error(f"Error updating with temporary data: {str(e)}")
    
    def update_with_complete_data(self, ticker, options_data, current_price):
        """Update the UI with complete data after loading is finished"""
        try:
            # Check if we have at least one expiration date with data
            if not options_data:
                logger.error(f"No options data available for {ticker}")
                self.status_label.config(text="No data available")
                self.show_loading_message("No options data available. Please try another ticker.")
                return
                
            logger.info(f"Creating OptionsDataProcessor for {ticker}")
            self.data_processor = OptionsDataProcessor(options_data, current_price)
            ds = self.data_processor.get_data()
            
            # Check if dataset is None or has no data
            if ds is None or len(ds.variables) == 0:
                logger.error(f"No options data available for {ticker}")
                self.status_label.config(text="No data available")
                self.show_loading_message("No options data available. Please try another ticker.")
                return
                
            # Get all expiration dates
            logger.info("Getting expiration dates")
            self.expiry_dates = self.data_processor.get_expirations()
            if not self.expiry_dates:
                logger.error(f"No expiration dates found for {ticker}")
                self.status_label.config(text="No expiry dates")
                self.show_loading_message("No expiration dates found. Please try another ticker.")
                return
            
            # Default to first expiry date (lowest DTE)
            self.current_expiry_index = 0
            logger.info(f"Setting to lowest DTE expiry: {self.expiry_dates[0]}")
            
            # Update UI
            self.update_expiry_display()
            self.update_plot()
            
            # Store the current ticker and update time
            self.current_ticker = ticker
            self.last_update_time = time.time()
            current_time = time.strftime("%H:%M:%S", time.localtime(self.last_update_time))
            self.status_label.config(text=f"Updated: {current_time}")
            
            # Mark data as loaded and schedule a refresh to ensure plot is updated
            self.data_loaded = True
            self.after(500, self.force_refresh_plot)
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error updating with complete data: {error_msg}")
            self.status_label.config(text=f"Error: {error_msg[:20]}...")
            
    def clean_ticker(self, ticker):
        """Clean and validate ticker input"""
        logger.info(f"Cleaning ticker input: {ticker}")
        ticker = ticker.strip().upper()
        if not ticker or not ticker.isalnum():
            return None
        logger.info(f"Cleaned ticker result: {ticker}")
        return ticker
    
    def search_ticker(self):
        ticker = self.clean_ticker(self.ticker_entry.get())
        if not ticker:
            messagebox.showwarning("Warning", 
                "Invalid ticker symbol. Please enter a valid stock symbol.")
            return
            
        logger.info(f"Searching for ticker: {ticker}")
        self.search_button.config(state='disabled')
        self.config(cursor="watch")
        self.show_loading_message(f"Fetching data for {ticker}...")
        
        # Update status to show loading
        self.status_label.config(text="Loading data...")
        
        # Reset data loaded flag
        self.data_loaded = False
        
        # Set a timeout for the data fetching process
        fetch_timeout = 45000  # 45 seconds in milliseconds
        self.fetch_timeout_id = self.after(fetch_timeout, self.handle_fetch_timeout)
        
        try:
            # Fetch data from Yahoo Finance with progressive loading
            logger.info(f"Fetching options data for {ticker}")
            options_data, current_price = self.api.get_options_data(ticker, 
                                                                   self.update_with_partial_data,
                                                                   max_dates=8)  # Limit to 8 expiration dates for faster loading
            
            # Cancel the timeout since we got a response
            self.after_cancel(self.fetch_timeout_id)
            
            if options_data is None or current_price is None:
                logger.error(f"Failed to fetch data for {ticker}")
                messagebox.showerror("Error", 
                    f"Failed to fetch data for {ticker}. Please try again in a few moments.")
                self.hide_loading_message()
                self.search_button.config(state='normal')
                self.config(cursor="")
                return
            
            # Final update with complete data
            self.update_with_complete_data(ticker, options_data, current_price)
            
        except Exception as e:
            # Cancel the timeout since we got an error
            self.after_cancel(self.fetch_timeout_id)
            
            error_msg = str(e)
            logger.error(f"Error in search_ticker: {error_msg}")
            if "Too Many Requests" in error_msg:
                messagebox.showerror("Rate Limit Error", 
                    "Yahoo Finance rate limit reached. Please try again in a few minutes.")
            else:
                messagebox.showerror("Error", 
                    f"An error occurred: {error_msg}")
        finally:
            self.search_button.config(state='normal')
            self.config(cursor="")
            self.hide_loading_message()
    
    def show_loading_message(self, message):
        """Show a loading message in the plot area"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, message, ha='center', va='center', fontsize=12)
        ax.set_axis_off()
        # Use draw_idle for better performance
        self.canvas.draw_idle()
    
    def hide_loading_message(self):
        """Clear the loading message"""
        self.figure.clear()
        # Use draw_idle for better performance
        self.canvas.draw_idle()
    
    def update_plot(self):
        """Update the plot with current data"""
        if not self.data_processor or not self.expiry_dates:
            logger.error("Cannot update plot: No data processor or expiry dates")
            self.show_loading_message("No data available for plotting")
            return
            
        try:
            current_date = self.expiry_dates[self.current_expiry_index]
            
            # Get data for the current expiration date using the new method
            df = self.data_processor.get_data_for_expiry(current_date)
            
            if df is None or df.empty:
                logger.error(f"No data available for expiry date {current_date}")
                self.show_loading_message(f"No data for {current_date.strftime('%Y-%m-%d')}")
                return
            
            # Clear previous plot
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            
            # Plot calls and puts separately
            display_field = self.plot_var.get()
            
            # Map display name to actual field name
            field_mapping = {
                "Spot": "spot",
                "Last Price": "lastPrice",
                "Bid": "bid",
                "Ask": "ask",
                "Volume": "volume",
                "Intrinsic Value": "intrinsic_value",
                "Extrinsic Value": "extrinsic_value",
            }
            
            # Get the database field name from the display name
            plot_field = field_mapping.get(display_field, display_field)
            
            calls = df[df['option_type'] == 'call']
            puts = df[df['option_type'] == 'put']
            
            # Create plot lines and store them for legend
            self.call_line = None
            self.put_line = None
            self.price_line = None
            
            if not calls.empty:
                self.call_line = ax.plot(calls['strike'], calls[plot_field], 'b-', label='Calls')[0]
            if not puts.empty:
                self.put_line = ax.plot(puts['strike'], puts[plot_field], 'r-', label='Puts')[0]
            
            # Add current price line if available
            if self.data_processor.current_price:
                self.price_line = ax.axvline(x=self.data_processor.current_price, color='g', 
                         linestyle='--', label=f'Current Price: ${self.data_processor.current_price:.2f}')
            
            # Get global min and max strike prices and set fixed x-axis limits
            min_strike, max_strike = self.data_processor.get_strike_range()
            if min_strike is not None and max_strike is not None:
                # Add a small buffer (5%) on each side for better visualization
                x_range = max_strike - min_strike
                buffer = x_range * 0.05
                ax.set_xlim(min_strike - buffer, max_strike + buffer)
            
            # Format title and labels
            expiry_date_str = current_date.strftime('%Y-%m-%d')
            # Get DTE for the current expiration date
            dte = df['DTE'].iloc[0] if not df.empty else "N/A"
            ax.set_title(f"{display_field} vs Strike Price - {expiry_date_str} (DTE: {dte})", fontsize=12, fontweight='bold')
            ax.set_xlabel('Strike Price ($)', fontsize=10)
            
            # Add dollar sign to y-axis label for price-related fields
            price_fields = ["Spot", "Last Price", "Bid", "Ask", "Intrinsic Value", "Extrinsic Value"]
            if display_field in price_fields:
                ax.set_ylabel(f'{display_field} ($)', fontsize=10)
            else:
                ax.set_ylabel(display_field, fontsize=10)
            
            # Create legend with initial entries
            self.original_legend_elements = []
            if self.call_line:
                self.original_legend_elements.append(self.call_line)
            if self.put_line:
                self.original_legend_elements.append(self.put_line)
            if self.price_line:
                self.original_legend_elements.append(self.price_line)
                
            ax.legend(handles=self.original_legend_elements)
            ax.grid(True)
            
            # Format x-axis with dollar signs
            def dollar_formatter(x, pos):
                return f'${x:.0f}'
                
            ax.xaxis.set_major_formatter(FuncFormatter(dollar_formatter))
            
            # Format y-axis with dollar signs for price fields
            if display_field in price_fields and display_field != "Volume":
                ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f'${y:.2f}'))
            
            # Add crosshair functionality
            # Create empty line objects for crosshairs
            self.h_line = ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7, visible=False)
            self.v_line = ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7, visible=False)
            
            # Create crosshair legend entries (initially hidden)
            self.strike_line = plt.Line2D([0], [0], color='gray', linestyle='--', alpha=0, label='Strike: $0.00')
            self.value_line = plt.Line2D([0], [0], color='gray', linestyle='--', alpha=0, label=f'{display_field}: $0.00')
            
            # Store references to important objects for the mouse motion event
            self.current_ax = ax
            self.current_display_field = display_field
            self.is_price_field = display_field in price_fields
            
            # Draw the canvas once to create the renderer
            self.canvas.draw()
            # Save the background for blitting
            self.background = self.canvas.copy_from_bbox(ax.bbox)
            
            # Connect the mouse motion event to the canvas
            self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
            
            logger.info(f"Plot updated successfully for {display_field}")
            
        except Exception as e:
            logger.error(f"Error updating plot: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            self.show_loading_message(f"Error updating plot: {str(e)[:50]}...")
    
    def on_mouse_move(self, event):
        """Handle mouse movement over the plot to update crosshairs"""
        try:
            # Only update if the mouse is over the plot area
            if event.inaxes == self.current_ax:
                # Get the current x and y values
                x, y = event.xdata, event.ydata
                
                # Throttle updates to reduce lag
                current_time = time.time()
                if hasattr(self, 'last_crosshair_update') and current_time - self.last_crosshair_update < 0.05:
                    # Skip this update if it's too soon after the last one (20 updates per second max)
                    return
                self.last_crosshair_update = current_time
                
                # Update crosshair positions
                self.h_line.set_ydata(y)
                self.v_line.set_xdata(x)
                self.h_line.set_visible(True)
                self.v_line.set_visible(True)
                
                # Format the values for display
                if self.is_price_field:
                    y_text = f'${y:.2f}'
                else:
                    y_text = f'{y:.0f}'
                
                # Update the crosshair legend entries
                self.strike_line.set_alpha(1)
                self.value_line.set_alpha(1)
                self.strike_line.set_label(f'Strike: ${x:.2f}')
                self.value_line.set_label(f'{self.current_display_field}: {y_text}')
                
                # Update the legend with crosshair values
                legend_elements = self.original_legend_elements.copy()
                legend_elements.extend([self.strike_line, self.value_line])
                self.current_ax.legend(handles=legend_elements)
                
                # Use blit=True for faster rendering of specific artists
                self.figure.canvas.restore_region(self.background)
                self.current_ax.draw_artist(self.h_line)
                self.current_ax.draw_artist(self.v_line)
                self.figure.canvas.blit(self.current_ax.bbox)
            else:
                # Hide crosshairs when mouse leaves plot area
                self.h_line.set_visible(False)
                self.v_line.set_visible(False)
                
                # Reset legend to original
                self.current_ax.legend(handles=self.original_legend_elements)
                self.canvas.draw_idle()
                
        except Exception as e:
            logger.error(f"Error in crosshair update: {str(e)}")
            # Don't show error to user for crosshair issues
            
    def force_refresh_plot(self):
        """Force a refresh of the plot to ensure data is displayed correctly"""
        logger.info("Forcing plot refresh to ensure data is displayed")
        if self.data_loaded and hasattr(self, 'data_processor') and self.expiry_dates:
            self.update_plot()
        
    def handle_fetch_timeout(self):
        """Handle timeout during data fetching"""
        logger.warning("Data fetch timeout reached")
        self.search_button.config(state='normal')
        self.config(cursor="")
        messagebox.showwarning("Timeout", 
            "Data fetching is taking too long. This may be due to rate limiting or network issues.\n\n"
            "If partial data was loaded, you can still view it, or try again later.")
        
        # If we have partial data, try to display it
        if hasattr(self, 'temp_data_processor') and hasattr(self, 'temp_expiry_dates'):
            logger.info("Using partial data after timeout")
            self.data_processor = self.temp_data_processor
            self.expiry_dates = self.temp_expiry_dates
            self.current_expiry_index = 0
            self.update_expiry_display()
            self.update_plot()
            
            # Store the current ticker and update time
            self.current_ticker = self.ticker_entry.get()
            self.last_update_time = time.time()
            current_time = time.strftime("%H:%M:%S", time.localtime(self.last_update_time))
            self.status_label.config(text=f"Partial data updated: {current_time}")
            
            # Mark data as loaded
            self.data_loaded = True

if __name__ == "__main__":
    app = OptionsVisualizerApp()
    app.mainloop()
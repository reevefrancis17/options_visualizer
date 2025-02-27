# offline_app.py
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib
# Force the TkAgg backend for better event handling with Tkinter
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.ticker import FuncFormatter
import pandas as pd
import numpy as np
import logging
import time
import os
import threading
import sys
# Add the parent directory to sys.path to enable absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from python.options_data import OptionsDataManager, OptionsDataProcessor
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
    # Field mapping for plot display
    FIELD_MAPPING = {
        "Spot": "spot",
        "Bid": "bid",
        "Ask": "ask",
        "Volume": "volume",
        "Intrinsic Value": "intrinsic_value",
        "Extrinsic Value": "extrinsic_value",
    }
    
    # Price-related fields that need dollar formatting
    PRICE_FIELDS = ["Spot", "Bid", "Ask", "Intrinsic Value", "Extrinsic Value"]
    
    def __init__(self):
        super().__init__()
        self.title("Options Visualizer")
        self.geometry("1200x800")  # Increased size for better visibility
        self.minsize(1000, 600)    # Set minimum window size
        
        logger.info("Initializing Options Visualizer App")
        self.api = OptionsDataManager()
        self.data_processor = None
        self.expiry_dates = []
        self.current_expiry_index = 0
        self.current_ticker = ""
        self.last_update_time = 0
        self.data_loaded = False  # Track if data has been loaded
        
        # Initialize crosshair variables
        self.h_line_call = None  # Horizontal line for calls
        self.h_line_put = None   # Horizontal line for puts
        self.v_line = None       # Vertical line for strike
        self.strike_line = None
        self.value_line = None
        self.call_value_line = None
        self.put_value_line = None
        self.background = None
        self.original_legend_elements = []
        self.current_ax = None
        self.current_display_field = None
        self.is_price_field = False
        self.last_crosshair_update = 0
        
        self.create_widgets()
        
        # Load default ticker immediately
        self.load_default_ticker()
        
    def create_widgets(self):
        """Create and arrange all UI widgets"""
        # Top frame for ticker input and navigation
        top_frame = tk.Frame(self)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        # Create ticker input section
        self._create_ticker_section(top_frame)
        
        # Create expiry navigation section
        self._create_navigation_section(top_frame)
        
        # Create main container for plot
        main_container = tk.Frame(self)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create plot options
        self._create_plot_options(main_container)
        
        # Create matplotlib figure and canvas
        self.figure = plt.Figure(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.figure, main_container)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Bind Enter key to search function
        self.bind('<Return>', lambda event: self.search_ticker())
    
    def _create_ticker_section(self, parent):
        """Create the ticker input section"""
        ticker_frame = tk.Frame(parent)
        ticker_frame.pack(side=tk.LEFT)
        
        tk.Label(ticker_frame, text="Ticker:").pack(side=tk.LEFT)
        self.ticker_entry = tk.Entry(ticker_frame, width=15)
        self.ticker_entry.pack(side=tk.LEFT, padx=5)
        self.search_button = tk.Button(ticker_frame, text="Search", command=self.search_ticker)
        self.search_button.pack(side=tk.LEFT)
        
        # Status label for last update time
        self.status_label = tk.Label(ticker_frame, text="Not updated yet", font=("Arial", 8))
        self.status_label.pack(side=tk.LEFT, padx=10)
    
    def _create_navigation_section(self, parent):
        """Create the expiry date navigation section"""
        nav_frame = tk.Frame(parent)
        nav_frame.pack(side=tk.RIGHT, padx=20)
        
        self.prev_button = tk.Button(nav_frame, text="◀", command=self.prev_expiry, state=tk.DISABLED)
        self.prev_button.pack(side=tk.LEFT, padx=5)
        
        self.expiry_label = tk.Label(nav_frame, text="No data", width=12)
        self.expiry_label.pack(side=tk.LEFT, padx=5)
        
        self.next_button = tk.Button(nav_frame, text="▶", command=self.next_expiry, state=tk.DISABLED)
        self.next_button.pack(side=tk.LEFT, padx=5)
    
    def _create_plot_options(self, parent):
        """Create the plot type selection options"""
        self.plot_var = tk.StringVar(value="Spot")
        plot_options_frame = ttk.LabelFrame(parent, text="Plot Type", padding=5)
        plot_options_frame.pack(fill=tk.X, pady=5)
        
        # Use the keys from FIELD_MAPPING for plot options
        for option in self.FIELD_MAPPING.keys():
            ttk.Radiobutton(
                plot_options_frame, 
                text=option, 
                value=option, 
                variable=self.plot_var, 
                command=self.update_plot
            ).pack(side=tk.LEFT, padx=5)
    
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
        """Update the expiry date display and navigation buttons"""
        if not self.expiry_dates:
            self.expiry_label.config(text="No data")
            self.prev_button.config(state=tk.DISABLED)
            self.next_button.config(state=tk.DISABLED)
            return
            
        # Get current date and update label
        current_date = self.expiry_dates[self.current_expiry_index]
        self.expiry_label.config(text=current_date.strftime('%Y-%m-%d'))
        
        # Update navigation buttons
        self.prev_button.config(state=tk.NORMAL if self.current_expiry_index > 0 else tk.DISABLED)
        self.next_button.config(state=tk.NORMAL if self.current_expiry_index < len(self.expiry_dates) - 1 else tk.DISABLED)
        
        # Log navigation state
        logger.info(f"Updated expiry display: {current_date.strftime('%Y-%m-%d')} " +
                   f"({self.current_expiry_index + 1}/{len(self.expiry_dates)})")
    
    def load_default_ticker(self):
        """Load the default ticker (SPY) on startup"""
        logger.info("Loading default ticker: SPY")
        self.ticker_entry.insert(0, "SPY")
        
        # Update status to show loading
        self.status_label.config(text="Loading SPY data...")
        
        # Load data immediately
        self._load_default_ticker_data()
    
    def _load_default_ticker_data(self):
        """Actually load the data for the default ticker"""
        ticker = "SPY"
        # Update status to show loading
        self.status_label.config(text="Loading SPY data...")
        
        # Use threading for data fetching
        threading.Thread(target=self._fetch_data, args=(ticker,), daemon=True).start()
    
    def update_with_partial_data(self, partial_data, current_price, processed_dates, total_dates):
        """Update the UI with partial data as it's being loaded"""
        # Handle different stages of data loading
        if not current_price:
            self.status_label.config(text="Fetching price...")
            return
            
        if not partial_data:
            self.status_label.config(text=f"Price: ${current_price:.2f} - Fetching data...")
            return
            
        # Calculate progress percentage
        progress_pct = int((processed_dates / total_dates) * 100) if total_dates > 0 else 0
        
        # Update status label with progress
        self.status_label.config(text=f"Loading: {progress_pct}% ({processed_dates}/{total_dates})")
        
        try:
            # The OptionsDataManager will create a processor with the partial data
            # We don't need to create it here, but we still need to check if we have data
            if processed_dates == 0:
                return
                
            # If this is the first time we're getting data and we have enough to display,
            # we can request a full update from the data manager in the next cycle
            if not hasattr(self, 'data_processor') or not self.data_loaded:
                if processed_dates >= 1:  # If we have at least one date processed
                    # Update status to show we're preparing to display data
                    self.status_label.config(text=f"Preparing data ({processed_dates}/{total_dates})...")
            
            # Update status with progress
            self.status_label.config(text=f"Loading: {progress_pct}% ({processed_dates}/{total_dates})")
            
        except Exception as e:
            logger.error(f"Error processing partial data: {str(e)}")
            # Continue loading - don't interrupt the process for partial data errors
    
    def update_with_complete_data(self, ticker, processor):
        """Update the UI with complete data after loading is finished"""
        try:
            # Reset UI state at the end regardless of outcome
            def reset_ui():
                self.search_button.config(state='normal')
                self.config(cursor="")
            
            # Check if we have data
            if not processor:
                logger.error(f"No options data available for {ticker}")
                self.status_label.config(text="No data available")
                reset_ui()
                return
                
            logger.info(f"Using OptionsDataProcessor for {ticker}")
            self.data_processor = processor
            ds = self.data_processor.get_data()
            
            # Check if dataset is valid
            if ds is None or len(ds.variables) == 0:
                logger.error(f"No options data available for {ticker}")
                self.status_label.config(text="No data available")
                reset_ui()
                return
                
            # Get all expiration dates
            logger.info("Getting expiration dates")
            self.expiry_dates = self.data_processor.get_expirations()
            if not self.expiry_dates:
                logger.error(f"No expiration dates found for {ticker}")
                self.status_label.config(text="No expiry dates")
                reset_ui()
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
            
            # Update status label with completion info
            total_dates = len(self.expiry_dates)
            self.status_label.config(text=f"Updated: {current_time} | {total_dates} dates loaded")
            
            # Mark data as loaded and refresh plot immediately
            self.data_loaded = True
            self.force_refresh_plot()
            
            # Reset UI state
            reset_ui()
            
        except Exception as e:
            logger.error(f"Error updating with complete data: {str(e)}")
            self.status_label.config(text=f"Error: {str(e)[:30]}...")
            self.search_button.config(state='normal')
            self.config(cursor="")
    
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
        
        # Update status to show loading
        self.status_label.config(text=f"Loading {ticker} data...")
        
        # Reset data loaded flag
        self.data_loaded = False
        
        # Use threading for data fetching
        threading.Thread(target=self._fetch_data, args=(ticker,), daemon=True).start()
    
    def _fetch_data(self, ticker):
        """Fetch data in a separate thread to keep UI responsive"""
        try:
            # Record start time for progress estimation
            self.last_update_time = time.time()
            
            # Fetch data using the data manager with progressive loading
            logger.info(f"Fetching options data for {ticker}")
            processor, current_price = self.api.get_options_data(ticker, 
                                                               progress_callback=self.update_with_partial_data)
            
            # Use after to update UI from the main thread
            if processor is None or current_price is None:
                logger.error(f"Failed to fetch data for {ticker}")
                self.after(0, lambda: self._handle_fetch_error(ticker, "Failed to fetch data"))
                return
            
            # Final update with complete data (using after to ensure it runs in the main thread)
            self.after(0, lambda: self.update_with_complete_data(ticker, processor))
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error in _fetch_data: {error_msg}")
            self.after(0, lambda: self._handle_fetch_error(ticker, error_msg))
    
    def _handle_fetch_error(self, ticker, error_msg):
        """Handle errors from data fetching"""
        # Update status label with error info
        self.status_label.config(text=f"Error: {error_msg[:30]}...")
        
        # Show appropriate error message
        if "Too Many Requests" in error_msg:
            messagebox.showerror("Rate Limit Error", 
                "Yahoo Finance rate limit reached. Please try again in a few minutes.")
        else:
            messagebox.showerror("Error", 
                f"An error occurred while fetching data for {ticker}: {error_msg}")
        
        # Try to get any cached data from the data manager
        try:
            # Check if we have a processor in the data manager's cache
            processor, current_price = self.api.get_options_data(ticker)
            
            if processor is not None:
                logger.info("Using cached data after error")
                
                # Use the processor
                self.data_processor = processor
                self.expiry_dates = self.data_processor.get_expirations()
                self.current_expiry_index = 0
                
                # Update UI
                self.update_expiry_display()
                self.update_plot()
                
                # Enable navigation if we have at least 2 dates
                if len(self.expiry_dates) >= 2:
                    self.next_button.config(state=tk.NORMAL)
                
                # Store the current ticker and update time
                self.current_ticker = ticker
                self.last_update_time = time.time()
                current_time = time.strftime("%H:%M:%S", time.localtime(self.last_update_time))
                
                # Update status with partial data info
                total_dates = len(self.expiry_dates)
                self.status_label.config(text=f"Cached data: {current_time} | {total_dates} dates")
                
                # Mark data as loaded
                self.data_loaded = True
        except Exception as e:
            logger.error(f"Error trying to use cached data: {str(e)}")
        
        # Reset UI state
        self.search_button.config(state='normal')
        self.config(cursor="")
    
    @staticmethod
    def dollar_formatter(x, pos):
        """Format a number as a dollar amount"""
        return f'${x:.0f}'
        
    def update_plot(self):
        """Update the plot with current data"""
        if not self.data_processor or not self.expiry_dates:
            logger.error("Cannot update plot: No data processor or expiry dates")
            self.status_label.config(text="No data available for plotting")
            return
            
        try:
            current_date = self.expiry_dates[self.current_expiry_index]
            df = self.data_processor.get_data_for_expiry(current_date)
            
            if df is None or df.empty:
                logger.error(f"No data available for expiry date {current_date}")
                self.status_label.config(text=f"No data for {current_date.strftime('%Y-%m-%d')}")
                return
            
            # Clear previous plot
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            
            display_field = self.plot_var.get()
            plot_field = self.FIELD_MAPPING.get(display_field, display_field)
            
            calls = df[df['option_type'] == 'call']
            puts = df[df['option_type'] == 'put']
            
            # Plot lines with explicit labels
            self.call_line = ax.plot(calls['strike'], calls[plot_field], 'b-', label='Calls: --')[0] if not calls.empty else None
            self.put_line = ax.plot(puts['strike'], puts[plot_field], 'r-', label='Puts: --')[0] if not puts.empty else None
            self.price_line = ax.axvline(x=self.data_processor.current_price, color='g', 
                                       linestyle='--', label=f'Spot: ${self.data_processor.current_price:.2f}') if self.data_processor.current_price else None
            
            # Set x-axis limits
            min_strike, max_strike = self.data_processor.get_strike_range()
            if min_strike is not None and max_strike is not None:
                x_range = max_strike - min_strike
                buffer = x_range * 0.05
                ax.set_xlim(min_strike - buffer, max_strike + buffer)
            
            # Set titles and labels
            expiry_date_str = current_date.strftime('%Y-%m-%d')
            today = pd.Timestamp.now().normalize()
            dte = max(0, (current_date - today).days)
            
            ax.set_title(f"{display_field} vs Strike Price - {expiry_date_str} (DTE: {dte})", fontsize=12, fontweight='bold')
            ax.set_xlabel('Strike Price ($)', fontsize=10)
            ax.set_ylabel(f'{display_field} ($)' if display_field in self.PRICE_FIELDS else display_field, fontsize=10)
            
            # Create single legend
            handles = [h for h in [self.call_line, self.put_line, self.price_line] if h is not None]
            self.legend = ax.legend(handles=handles, loc='upper right')
            
            # Format axes
            ax.grid(True)
            ax.xaxis.set_major_formatter(FuncFormatter(self.dollar_formatter))
            if display_field in self.PRICE_FIELDS and display_field != "Volume":
                ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f'${y:.2f}'))
            
            # Initialize crosshairs
            self.h_line_call = ax.axhline(y=0, color='blue', linestyle='-', alpha=0.6, lw=1.0, visible=False)
            self.h_line_put = ax.axhline(y=0, color='red', linestyle='-', alpha=0.6, lw=1.0, visible=False)
            self.v_line = ax.axvline(x=0, color='gray', linestyle='-', alpha=0.8, lw=1.0, visible=False)
            
            # Store references
            self.current_ax = ax
            self.current_display_field = display_field
            self.is_price_field = display_field in self.PRICE_FIELDS
            
            # Pre-compute data for mouse movement
            self._precompute_mouse_data(df, plot_field)
            
            # Initial draw and background capture
            self.canvas.draw()
            self.background = self.canvas.copy_from_bbox(ax.bbox)
            
            # Connect events (ensure single connection)
            self.canvas.mpl_disconnect(self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move))
            self.canvas.mpl_disconnect(self.canvas.mpl_connect('figure_leave_event', self.on_mouse_leave))
            self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
            self.canvas.mpl_connect('figure_leave_event', self.on_mouse_leave)
            
            logger.info(f"Plot updated successfully for {display_field}")
            
        except Exception as e:
            logger.error(f"Error updating plot: {str(e)}")
            self.status_label.config(text=f"Error updating plot: {str(e)[:30]}...")

    def _precompute_mouse_data(self, df, plot_field):
        """Pre-compute data for faster mouse movement handling"""
        calls = df[df['option_type'] == 'call']
        puts = df[df['option_type'] == 'put']
        
        self.call_data = (calls['strike'].values, calls[plot_field].values) if not calls.empty else (np.array([]), np.array([]))
        self.put_data = (puts['strike'].values, puts[plot_field].values) if not puts.empty else (np.array([]), np.array([]))
        
        # Create lookup dictionaries
        self.call_lookup = dict(zip(self.call_data[0], self.call_data[1]))
        self.put_lookup = dict(zip(self.put_data[0], self.put_data[1]))

    def on_mouse_move(self, event):
        """Handle mouse movement with optimized performance"""
        # Throttle to 2ms (500 FPS max)
        current_time = time.time()
        if current_time - getattr(self, 'last_crosshair_update', 0) < 0.002:
            return
        self.last_crosshair_update = current_time
        
        if event.inaxes != self.current_ax or event.xdata is None:
            self._reset_crosshairs()
            return
            
        x = event.xdata
        
        # Find nearest strike using binary search
        call_strikes, call_values = self.call_data
        if len(call_strikes) > 0:
            idx = np.searchsorted(call_strikes, x)
            idx = min(max(0, idx-1), len(call_strikes)-1) if idx >= len(call_strikes) else idx
            nearest_x = call_strikes[idx]
            call_value = call_values[idx]
            
            # Direct lookup for put value
            put_value = self.put_lookup.get(nearest_x)
            
            # Format values
            fmt = f'${{:.2f}}' if self.is_price_field else '{:.0f}'
            call_text = fmt.format(call_value) if call_value is not None else 'N/A'
            put_text = fmt.format(put_value) if put_value is not None else 'N/A'
            
            # Update crosshairs
            self.v_line.set_xdata([nearest_x])
            self.h_line_call.set_ydata([call_value])
            self.h_line_put.set_ydata([put_value]) if put_value is not None else None
            
            self.v_line.set_visible(True)
            self.h_line_call.set_visible(True)
            self.h_line_put.set_visible(put_value is not None)
            
            # Update legend (single instance)
            texts = self.legend.get_texts()
            if len(texts) > 0:
                texts[0].set_text(f'Calls: {call_text}')
            if len(texts) > 1:
                texts[1].set_text(f'Puts: {put_text}')
            
            # Optimized redraw with blitting
            self.canvas.restore_region(self.background)
            self.current_ax.draw_artist(self.v_line)
            self.current_ax.draw_artist(self.h_line_call)
            if put_value is not None:
                self.current_ax.draw_artist(self.h_line_put)
            self.current_ax.draw_artist(self.legend)
            self.canvas.blit(self.current_ax.bbox)
        
    def on_mouse_leave(self, event):
        """Reset crosshairs and legend when mouse leaves"""
        self._reset_crosshairs()

    def _reset_crosshairs(self):
        """Reset crosshairs and legend to default state"""
        self.v_line.set_visible(False)
        self.h_line_call.set_visible(False)
        self.h_line_put.set_visible(False)
        
        texts = self.legend.get_texts()
        if len(texts) > 0:
            texts[0].set_text('Calls: --')
        if len(texts) > 1:
            texts[1].set_text('Puts: --')
        
        self.canvas.restore_region(self.background)
        self.current_ax.draw_artist(self.legend)
        self.canvas.blit(self.current_ax.bbox)

    def force_refresh_plot(self):
        """Force a refresh of the plot to ensure data is displayed correctly"""
        logger.info("Forcing plot refresh to ensure data is displayed")
        if self.data_loaded and hasattr(self, 'data_processor') and self.expiry_dates:
            self.update_plot()
        

if __name__ == "__main__":
    app = OptionsVisualizerApp()
    app.mainloop()
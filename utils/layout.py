# Import necessary packages
from kivy import platform
from kivy.properties import ObjectProperty
from kivy.uix.screenmanager import Screen
from kivy.core.window import Window
from kivy.uix.floatlayout import FloatLayout

# Define a custom widget for loading files
class LoadFile(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)
        
# Define the main widget for the app
class MainScreen(Screen):
    pass       


light_red_color = {
            "Red": {
                "50": "#F63D54",
                "100": "#F63D54",
                "200": "#F63D54",
                "300": "#F63D54",
                "400": "#F63D54",
                "500": "#F63D54",
                "600": "#F63D54",
                "700": "#F63D54",
                "800": "#F63D54",
                "900": "#F63D54",
                "A100": "#F63D54",
                "A200": "#F63D54",
                "A400": "#F63D54",
                "A700": "#F63D54",
                "ContrastText": "#F63D54",  # Text color for light theme
                "ContrastDark": "#F63D54"   # Text color for dark theme
            }
        }

if platform == "android":

    from jnius import autoclass
    from android.runnable import run_on_ui_thread
    from android import mActivity
    View = autoclass('android.view.View')

    @run_on_ui_thread
    def hide_landscape_status_bar(instance, width, height):
        # width,height gives false layout events, on pinch/spread 
        # so use Window.width and Window.height
        if Window.width > Window.height: 
            # Hide status bar
            option = View.SYSTEM_UI_FLAG_FULLSCREEN
        else:
            # Show status bar 
            option = View.SYSTEM_UI_FLAG_VISIBLE
        mActivity.getWindow().getDecorView().setSystemUiVisibility(option)
elif platform != 'ios':    
    # Dispose of that nasty red dot, required for gestures4kivy.
    from kivy.config import Config 
    Config.set('input', 'mouse', 'mouse, disable_multitouch')
    

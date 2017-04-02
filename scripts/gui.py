import _thread
import tensorflow
import keras
from functools import partial
from firebase import firebase
import multiprocessing
import kivy
kivy.require('1.9.2')
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.image import Image,AsyncImage
from kivy.uix.popup import Popup
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.gridlayout import GridLayout
from kivy.graphics import *
from kivy.graphics.instructions import Canvas
from kivy.core.window import Window
from kivy.clock import Clock
import gpsGraph
import random

Window.clearcolor = (1,1,1,1)

multiprocessing.freeze_support()

firebase = firebase.FirebaseApplication('https://sihuser-11acb.firebaseio.com/')

class login(Screen):
    def __init__(self,**kwargs):
        super(login, self).__init__(**kwargs)
        self.add_widget(Image(source='appLogo.png', size_hint=(0.17, 0.17), pos_hint={'x': 0.8, 'y': 0.83}))
        self.add_widget(Image(source='minofsteel.png',size_hint=(0.31,0.15),pos_hint={'x':0.02,'y':0.83}))
        self.add_widget(Label(text="[b]Username[/b]",color=(0,0,0,0.5),markup=True,font_size='18sp',size_hint=(0.1,0.1),pos_hint={'x':0.3,'y':0.5}))
        self.un = TextInput(multiline=False,size_hint=(0.3,0.05),pos_hint={'x':0.5,'y':0.53},write_tab=False,suggestion_text='admin')
        self.add_widget(self.un)
        self.add_widget(Label(text="[b]Password[/b]",color=(0,0,0,0.5),markup=True,font_size='18sp',size_hint=(0.1,0.1),pos_hint={'x':0.3,'y':0.4}))
        self.pw = TextInput(multiline=False,password=True,size_hint=(0.3,0.05),pos_hint={'x':0.5,'y':0.43},write_tab=False)
        self.add_widget(self.pw)
        self.loginButton = Button(text='[b]Login[/b]',markup=True,size_hint=(0.1,0.07),pos_hint={'x':0.4,'y':0.29})
        self.add_widget(self.loginButton)
        self.errorLabel = Label(text='Incorrect username/password. Try again.', font_size='16sp', size_hint=(0.1, 0.07),
                           pos_hint={'x': 0.4, 'y': 0.2})

        def chkAccess(dt):
            UN = str(self.un.text.strip())
            PW = str(self.pw.text)
            if UN == 'admin' and PW == 'admin':
                login.remove_widget(self, self.errorLabel)
                self.un.text = ''
                self.pw.text = ''
                sm.current = 'info'
            else:
                try:
                    self.add_widget(self.errorLabel)
                    self.pw.text = ''
                except:
                    pass

        self.loginButton.bind(on_press=chkAccess)

class info(Screen):
    def __init__(self,**kwargs):
        super(info, self).__init__(**kwargs)
        # self.canvas.add(Rectangle(source='softwareBackground.jpg',size=(1500,800),pos=(0,0)))
        # self.canvas.add(Rectangle(size=(800,80),pos=(0,400),Color=(1,0,0)))
        # self.add_widget(Image(source='softwareBackground.jpg', size_hint=(1,1), pos_hint=(1,1)))
        self.add_widget(Image(source='appLogo.png', size_hint=(0.17, 0.17), pos_hint={'x': 0.8, 'y': 0.83}))
        self.add_widget(Image(source='minofsteel.png', size_hint=(0.31, 0.15), pos_hint={'x': 0.02, 'y': 0.83}))
        logoutButton = Button(text='[b]Logout[/b]', markup=True, size_hint=(0.2,0.06), pos_hint={'x':0.84, 'y':0.72})
        self.add_widget(logoutButton)
        def loginPage(dt):
            sm.current = 'login'
        logoutButton.bind(on_press=loginPage)
        def switch2Open(dt):
            pass
        def switch2Pending(dt):
            pass
        def switch2Closed(dt):
            pass
        openSwitch = Button(text='Open Cases',size_hint=(0.28,0.06),pos_hint={'x':0,'y':0.72},on_press=switch2Pending)
        pendingSwitch = Button(text='Pending Cases',size_hint=(0.28,0.06),pos_hint={'x':0.28,'y':0.72},on_press=switch2Pending)
        closedSwitch = Button(text='Closed Cases',size_hint=(0.28,0.06),pos_hint={'x':0.56,'y':0.72},on_press=switch2Pending)
        self.add_widget(openSwitch)
        self.add_widget(pendingSwitch)
        self.add_widget(closedSwitch)

        self.infoLabel = Label(text='', font_size=23,size_hint=(0.5, 0.5), pos_hint={'x': 0.5, 'y': 0.29}, color=(1, 0, 0, 1))
        self.add_widget(self.infoLabel)
        self.theftIm = Image(size_hint=(0.01,0.01),pos_hint={'x':0.6,'y':0})

        def checkAndRead(parent='complaints'):
            result = firebase.get(parent, None)
            return result

        

        def selectComplaint(i,dt):
            i = (list(self.fn.keys())[int(i)])
            self.complaintID = i
            aadhar = self.fn[i]['Aadhar']
            address = self.fn[i]['Address']
            self.complaintAddress = address
            gps = self.fn[i]['GPS']
            image = self.fn[i]['image']
            name = self.fn[i]['Name']
            status = self.fn[i]['Status']
            comments = self.fn[i]['Comments']
            s = "Status : "+status+"\n"+"Aadhar UID : "+str(aadhar)+"\n"+"Name : "+str(name)+"\n"+"GPS : "+str(gps)+"\n"+"Comments : "+str(comments)
            self.infoLabel.text = s
            self.theftIm.source = image

            # listRange = [2,90,120,170,40,15]
            # v = random.randint(0,5)
            # v = listRange[v]
            # print("    ------ ",v)
            # gpsGraph.getGPS('gps1',v)

            self.add_widget(Button(text="Graph & Report",size_hint=(0.2,0.15),pos_hint={'x':0.7,'y':0.1},on_press=popUpWindow))

        def popUpWindow(dt):
            # PopUp

            listRange = [200,90,130,170,40,15,36,204,68,12]
            v = random.randint(0,9)
            v = listRange[v]
            print("  TERI MA KI  ------ ",v)
            gpsGraph.getGPS('gps1',204)

            content = GridLayout(cols=2)
            content.add_widget(Image(source='Graph1.png',size_hint=(1,1),pos_hint={'x':0.3,'y':0.01}))
            self.graphCaption = Label(text=str(self.complaintAddress),size_hint=(0.2,0.1))
            content.add_widget(self.graphCaption)
            cancelButton = Button(text='Close',size_hint=(0.01,0.05))
            reportButton = Button(text='Report',size_hint=(0.01,0.05))
            content.add_widget(cancelButton)
            content.add_widget(reportButton)
            popup = Popup(title='Anomalies',content=content, size=(500,500),auto_dismiss=False)
            cancelButton.bind(on_press=popup.dismiss)
            reportButton.bind(on_press=reportFn)
            popup.open()

        def reportFn(dt):
            self.graphCaption.text += "\n" + "Report Sent." + "\n Status : Pending" 
            firebase.patch("https://sihuser-11acb.firebaseio.com/complaints/"+self.complaintID,{'Status':'Pending'})

        def makeInitialComplaintLabels(complaintsNum):
            b = []
            Xcoor = 0.005
            Ycoor = 0.52
            for i in range(complaintsNum):
                b.append(Button(text='',size_hint=(0.45, None),font_size=12, pos_hint={'x': Xcoor, 'y': Ycoor}))
                self.add_widget(b[i])
                Ycoor -= 0.17
            return b

        def complaintLabels(dt):
            self.fn = checkAndRead()
            self.keys = self.fn.keys()

            complaints = []

            for i in self.keys:
                complaints.append(self.fn[i]['Address'])
            # print (complaints,len(complaints))

            index=0
            b = makeInitialComplaintLabels(len(complaints))
            for i in complaints:
                b[index].text = i
                b[index].bind(on_press = partial(selectComplaint, index))
                index += 1

        complaintLabels(None)
        Clock.schedule_interval(complaintLabels,10)

        # self.add_widget(Image(source='softwareBackground.jpg',size_hint=(0.6,1.8),pos_hint={'x':0,'y':0}))
        # self.canvas.add(Rectangle(source='softwareBackground.jpg',size=(80,40),pos=self.pos,Translate={'xy':self.pos}))
        # complaintLabels = Label(text=complaint[index],size_hint=(0.2,0.8),pos_hint={'x':0.05,'y':0.1})
        # self.add_widget(complaintLabels)

sm = ScreenManager()
sm.add_widget(login(name='login'))
sm.add_widget(info(name='info'))

# sm.add_widget(login(name='login'))

class Vidyut(App):
    def build(self):
        return sm

Vidyut().run()
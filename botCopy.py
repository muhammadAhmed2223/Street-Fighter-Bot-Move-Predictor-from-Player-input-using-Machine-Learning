from command import Command
import numpy as np
from buttons import Buttons
import pandas as pd
from joblib import dump, load
from sklearn.linear_model import LinearRegression
class Bot:
    def __init__(self):
        #< - v + < - v - v + > - > + Y
        self.fire_code=["<","!<","v+<","!v+!<","v","!v","v+>","!v+!>",">+Y","!>+!Y"]
        self.exe_code = 0
        self.start_fire=True
        self.remaining_code=[]
        self.my_command = Command()
        self.buttn= Buttons()
        self.df = pd.DataFrame(columns=['timer','fight_result','has_round_started','is_round_over','Player1_ID','health','x_coord','y_coord','is_jumping','is_crouching','is_player_in_move','move_id','player1_buttons up','player1_buttons down','player1_buttons right','player1_buttons left','Y','B','X','A','L','R','Player2_ID','Player2health','Player2x_coord','Player2y_coord','Player2is_jumping','Player2is_crouching','Player2is_player_in_move','Player2move_id','player2_buttons up','player2_buttons down','player2_buttons right','player2_buttons left','Player2 Y','Player2 B','Player2 X','Player2 A','Player2 L','Player2 R'])
        self.model1 = load('Model/Player1.joblib')
        self.model2 = load('Model/Player2.joblib')

        
    def getCommand(self,diff,pred):
        commands = []
        if diff > 0:
            tempStr1 = ""
            tempStr2 = ""
            if pred[3] == 1:
                tempStr1 += "<"
                tempStr2 += "!<"
                if pred[0] == 1:
                    tempStr1 += "+^"
                    tempStr2 += "+!^"
                    if pred[8] == 1:
                        tempStr1 += "+L"
                        tempStr2 += "+!L"
                    elif pred[4] == 1:
                        tempStr1 += "+Y"
                        tempStr2 += "+!Y"
                    elif pred[7] == 1:
                        tempStr1 += "+A"
                        tempStr2 += "+!A"
                    elif pred[9] == 1:
                        tempStr1 += "+R"
                        tempStr2 += "+!R"
                    elif pred[5] == 1:
                        tempStr1 += "+B"
                        tempStr2 += "+!B" 
                elif pred[4] == 1:
                    tempStr1 += "+Y"
                    tempStr2 += "+!Y"    
            elif pred[2] == 1:
                tempStr1 += ">"
                tempStr2 += "!>"
                if pred[0] == 1:
                    tempStr1 += "+^"
                    tempStr2 += "+!^"
                    if pred[8] == 1:
                        tempStr1 += "+L"
                        tempStr2 += "+!L"
                    elif pred[4] == 1:
                        tempStr1 += "+Y"
                        tempStr2 += "+!Y"
                    elif pred[7] == 1:
                        tempStr1 += "+A"
                        tempStr2 += "+!A"
                    elif pred[9] == 1:
                        tempStr1 += "+R"
                        tempStr2 += "+!R"
                    elif pred[5] == 1:
                        tempStr1 += "+B"
                        tempStr2 += "+!B"
                elif pred[4] == 1:
                    tempStr1 += "+Y"
                    tempStr2 += "+!Y" 
            elif pred[0] == 1:
                tempStr1 += "^"
                tempStr2 += "!^"
            elif pred[1] == 1:
                tempStr1 += "v"
                tempStr2 += "!v"
                if pred[9] == 1:
                    tempStr1 += "+R"
                    tempStr2 += "+!R"
                elif pred[3] == 1:
                    tempStr1 += "+<"
                    tempStr2 += "+!<"
                elif pred[2] == 1:
                    tempStr1 += "+>"
                    tempStr2 += "+!>"
            commands.append(tempStr1)
            commands.append(tempStr2)
        else:
            tempStr1 = ""
            tempStr2 = ""
               
            if pred[2] == 1:
                tempStr1 += ">"
                tempStr2 += "!>"
                if pred[0] == 1:
                    tempStr1 += "+^"
                    tempStr2 += "+!^"
                    if pred[8] == 1:
                        tempStr1 += "+L"
                        tempStr2 += "+!L"
                    elif pred[4] == 1:
                        tempStr1 += "+Y"
                        tempStr2 += "+!Y"
                    elif pred[7] == 1:
                        tempStr1 += "+A"
                        tempStr2 += "+!A"
                    elif pred[9] == 1:
                        tempStr1 += "+R"
                        tempStr2 += "+!R"
                    elif pred[5] == 1:
                        tempStr1 += "+B"
                        tempStr2 += "+!B"
                elif pred[4] == 1:
                    tempStr1 += "+Y"
                    tempStr2 += "+!Y" 
            if pred[3] == 1:
                tempStr1 += "<"
                tempStr2 += "!<"
                if pred[0] == 1:
                    tempStr1 += "+^"
                    tempStr2 += "+!^"
                    if pred[8] == 1:
                        tempStr1 += "+L"
                        tempStr2 += "+!L"
                    elif pred[4] == 1:
                        tempStr1 += "+Y"
                        tempStr2 += "+!Y"
                    elif pred[7] == 1:
                        tempStr1 += "+A"
                        tempStr2 += "+!A"
                    elif pred[9] == 1:
                        tempStr1 += "+R"
                        tempStr2 += "+!R"
                    elif pred[5] == 1:
                        tempStr1 += "+B"
                        tempStr2 += "+!B" 
                elif pred[4] == 1:
                    tempStr1 += "+Y"
                    tempStr2 += "+!Y" 
            elif pred[0] == 1:
                tempStr1 += "^"
                tempStr2 += "!^"
            elif pred[1] == 1:
                tempStr1 += "v"
                tempStr2 += "!v"
                if pred[9] == 1:
                    tempStr1 += "+R"
                    tempStr2 += "+!R"
                elif pred[2] == 1:
                    tempStr1 += "+>"
                    tempStr2 += "+!>"
                elif pred[3] == 1:
                    tempStr1 += "+<"
                    tempStr2 += "+!<"
                
            commands.append(tempStr1)
            commands.append(tempStr2)
        return commands
        
    def getTrueFalse(self,item):
        if (item == 0):
            return False
        if (item == 1):
            return True
    def fight(self,current_game_state,player):
        #python Videos\gamebot-competition-master\PythonAPI\controller.py 1
        if player=="1":
            #print("1")
            #v - < + v - < + B spinning
            data = np.array([
            current_game_state.player1.x_coord,
            current_game_state.player1.y_coord,
            current_game_state.player2.x_coord,
            current_game_state.player2.y_coord,
            current_game_state.player2.is_jumping,
            current_game_state.player2.is_crouching,
            current_game_state.player2.is_player_in_move,
            current_game_state.player2.player_buttons.up,
            current_game_state.player2.player_buttons.down,
            current_game_state.player2.player_buttons.right,
            current_game_state.player2.player_buttons.left,
            current_game_state.player2.player_buttons.Y,
            current_game_state.player2.player_buttons.B,
            current_game_state.player2.player_buttons.X,
            current_game_state.player2.player_buttons.A,
            current_game_state.player2.player_buttons.L,
            current_game_state.player2.player_buttons.R])

            # create a list of column names
            columns = [ 'x_coord', 'y_coord', 'Player2x_coord','Player2y_coord','Player2is_jumping',
                        'Player2is_crouching','Player2is_player_in_move','player2_buttons up',
                        'player2_buttons down','player2_buttons right','player2_buttons left',
                        'Player2 Y','Player2 B','Player2 X',
                        'Player2 A','Player2 L','Player2 R'
                       ]

            # create a DataFrame
            df = pd.DataFrame(data.reshape(1, -1), columns=columns)
            pred = self.model1.predict(df)
            pred = np.round(pred)
            pred = np.absolute(pred)
            if( self.exe_code!=0  ):
                self.run_command([],current_game_state.player1)
            diff=current_game_state.player2.x_coord - current_game_state.player1.x_coord
            commands = self.getCommand(diff,pred[0])
            self.run_command(commands,current_game_state.player1)
            self.my_command.player_buttons=self.buttn




        elif player=="2":
            data = np.array([
            current_game_state.player1.x_coord,
            current_game_state.player1.y_coord,
            current_game_state.player2.x_coord,
            current_game_state.player2.y_coord,
            current_game_state.player1.is_jumping,
            current_game_state.player1.is_crouching,
            current_game_state.player1.is_player_in_move,
            current_game_state.player1.player_buttons.up,
            current_game_state.player1.player_buttons.down,
            current_game_state.player1.player_buttons.right,
            current_game_state.player1.player_buttons.left,
            current_game_state.player1.player_buttons.Y,
            current_game_state.player1.player_buttons.B,
            current_game_state.player1.player_buttons.X,
            current_game_state.player1.player_buttons.A,
            current_game_state.player1.player_buttons.L,
            current_game_state.player1.player_buttons.R])

            columns = [ 'x_coord', 'y_coord', 'Player2x_coord','Player2y_coord','is_jumping',
                        'is_crouching','is_player_in_move','player1_buttons up',
                        'player1_buttons down','player1_buttons right','player1_buttons left',
                        'Y','B','X','A','L','R'
                       ]

            # create a DataFrame
            df = pd.DataFrame(data.reshape(1, -1), columns=columns)
            pred = self.model2.predict(data)
            pred = np.round(pred)
            pred = np.absolute(pred)
            if( self.exe_code!=0 ):
               self.run_command([],current_game_state.player2)
            diff=current_game_state.player1.x_coord - current_game_state.player2.x_coord
            commands = self.getCommand(diff,pred[0])
            self.run_command(commands,current_game_state.player2)
            self.my_command.player2_buttons=self.buttn

            

            # append the new tuple to the dataframe
        new_tuple = (current_game_state.timer,current_game_state.fight_result,
            current_game_state.has_round_started,
            current_game_state.is_round_over,
            current_game_state.player1.player_id,
            current_game_state.player1.health,
            current_game_state.player1.x_coord,
            current_game_state.player1.y_coord,
            current_game_state.player1.is_jumping,
            current_game_state.player1.is_crouching,
            current_game_state.player1.is_player_in_move,
            current_game_state.player1.move_id,
            current_game_state.player1.player_buttons.up,
            current_game_state.player1.player_buttons.down,
            current_game_state.player1.player_buttons.right,
            current_game_state.player1.player_buttons.left,
            current_game_state.player1.player_buttons.Y,
            current_game_state.player1.player_buttons.B,
            current_game_state.player1.player_buttons.X,
            current_game_state.player1.player_buttons.A,
            current_game_state.player1.player_buttons.L,
            current_game_state.player1.player_buttons.R,
            current_game_state.player2.player_id,
            current_game_state.player2.health,
            current_game_state.player2.x_coord,
            current_game_state.player2.y_coord,
            current_game_state.player2.is_jumping,
            current_game_state.player2.is_crouching,
            current_game_state.player2.is_player_in_move,
            current_game_state.player2.move_id,
            current_game_state.player2.player_buttons.up,
            current_game_state.player2.player_buttons.down,
            current_game_state.player2.player_buttons.right,
            current_game_state.player2.player_buttons.left,
            current_game_state.player2.player_buttons.Y,
            current_game_state.player2.player_buttons.B,
            current_game_state.player2.player_buttons.X,
            current_game_state.player2.player_buttons.A,
            current_game_state.player2.player_buttons.L,
            current_game_state.player2.player_buttons.R)

        new_data = pd.DataFrame({
            'timer': [new_tuple[0]],
            'fight_result': [new_tuple[1]],
            'has_round_started': [new_tuple[2]],
            'is_round_over': [new_tuple[3]],
            'Player1_ID': [new_tuple[4]],
            'health': [new_tuple[5]],
            'x_coord': [new_tuple[6]],
            'y_coord': [new_tuple[7]],
            'is_jumping': [new_tuple[8]],
            'is_crouching': [new_tuple[9]],
            'is_player_in_move': [new_tuple[10]],
            'move_id': [new_tuple[11]],
            'player1_buttons up': [new_tuple[12]],
            'player1_buttons down': [new_tuple[13]],
            'player1_buttons right': [new_tuple[14]],
            'player1_buttons left': [new_tuple[15]],
            'Y': [new_tuple[16]],
            'B': [new_tuple[17]],
            'X': [new_tuple[18]],
            'A': [new_tuple[19]],
            'L': [new_tuple[20]],
            'R': [new_tuple[21]],
            'Player2_ID': [new_tuple[22]],
            'Player2health': [new_tuple[23]],
            'Player2x_coord': [new_tuple[24]],
            'Player2y_coord': [new_tuple[25]],
            'Player2is_jumping': [new_tuple[26]],
            'Player2is_crouching': [new_tuple[27]],
            'Player2is_player_in_move': [new_tuple[28]],
            'Player2move_id': [new_tuple[29]],
            'player2_buttons up': [new_tuple[30]],
            'player2_buttons down': [new_tuple[31]],
            'player2_buttons right': [new_tuple[32]],
            'player2_buttons left': [new_tuple[33]],
            'Player2 Y': [new_tuple[34]],
            'Player2 B': [new_tuple[35]],
            'Player2 X': [new_tuple[36]],
            'Player2 A': [new_tuple[37]],
            'Player2 L': [new_tuple[38]],
            'Player2 R': [new_tuple[39]]
        })
        self.df = self.df = pd.concat([self.df, new_data], ignore_index=True)
        if current_game_state.is_round_over != False:
            if current_game_state.fight_result == 'P2':
                self.df.to_csv('Model/Player2.csv', mode='a',header=False, index=False)
            elif current_game_state.fight_result == 'P1':
                self.df.to_csv('Model/Player1.csv', mode='a',header=False, index=False)

        return self.my_command



    def run_command( self , com , player   ):

        if self.exe_code-1==len(self.fire_code):
            self.exe_code=0
            self.start_fire=False
            print ("compelete")
            #exit()
            # print ( "left:",player.player_buttons.left )
            # print ( "right:",player.player_buttons.right )
            # print ( "up:",player.player_buttons.up )
            # print ( "down:",player.player_buttons.down )
            # print ( "Y:",player.player_buttons.Y )

        elif len(self.remaining_code)==0 :

            self.fire_code=com
            #self.my_command=Command()
            self.exe_code+=1

            self.remaining_code=self.fire_code[0:]

        else:
            self.exe_code+=1
            if self.remaining_code[0]=="v+<":
                self.buttn.down=True
                self.buttn.left=True
                print("v+<")
            elif self.remaining_code[0]=="!v+!<":
                self.buttn.down=False
                self.buttn.left=False
                print("!v+!<")
            elif self.remaining_code[0]=="v+>":
                self.buttn.down=True
                self.buttn.right=True
                print("v+>")
            elif self.remaining_code[0]=="!v+!>":
                self.buttn.down=False
                self.buttn.right=False
                print("!v+!>")

            elif self.remaining_code[0]==">+Y":
                self.buttn.Y= True #not (player.player_buttons.Y)
                self.buttn.right=True
                print(">+Y")
            elif self.remaining_code[0]=="!>+!Y":
                self.buttn.Y= False #not (player.player_buttons.Y)
                self.buttn.right=False
                print("!>+!Y")

            elif self.remaining_code[0]=="<+Y":
                self.buttn.Y= True #not (player.player_buttons.Y)
                self.buttn.left=True
                print("<+Y")
            elif self.remaining_code[0]=="!<+!Y":
                self.buttn.Y= False #not (player.player_buttons.Y)
                self.buttn.left=False
                print("!<+!Y")

            elif self.remaining_code[0]== ">+^+L" :
                self.buttn.right=True
                self.buttn.up=True
                self.buttn.L= not (player.player_buttons.L)
                print(">+^+L")
            elif self.remaining_code[0]== "!>+!^+!L" :
                self.buttn.right=False
                self.buttn.up=False
                self.buttn.L= False #not (player.player_buttons.L)
                print("!>+!^+!L")

            elif self.remaining_code[0]== ">+^+Y" :
                self.buttn.right=True
                self.buttn.up=True
                self.buttn.Y= not (player.player_buttons.Y)
                print(">+^+Y")
            elif self.remaining_code[0]== "!>+!^+!Y" :
                self.buttn.right=False
                self.buttn.up=False
                self.buttn.Y= False #not (player.player_buttons.L)
                print("!>+!^+!Y")


            elif self.remaining_code[0]== ">+^+R" :
                self.buttn.right=True
                self.buttn.up=True
                self.buttn.R= not (player.player_buttons.R)
                print(">+^+R")
            elif self.remaining_code[0]== "!>+!^+!R" :
                self.buttn.right=False
                self.buttn.up=False
                self.buttn.R= False #ot (player.player_buttons.R)
                print("!>+!^+!R")

            elif self.remaining_code[0]== ">+^+A" :
                self.buttn.right=True
                self.buttn.up=True
                self.buttn.A= not (player.player_buttons.A)
                print(">+^+A")
            elif self.remaining_code[0]== "!>+!^+!A" :
                self.buttn.right=False
                self.buttn.up=False
                self.buttn.A= False #not (player.player_buttons.A)
                print("!>+!^+!A")

            elif self.remaining_code[0]== ">+^+B" :
                self.buttn.right=True
                self.buttn.up=True
                self.buttn.B= not (player.player_buttons.B)
                print(">+^+B")
            elif self.remaining_code[0]== "!>+!^+!B" :
                self.buttn.right=False
                self.buttn.up=False
                self.buttn.B= False #not (player.player_buttons.A)
                print("!>+!^+!B")

            elif self.remaining_code[0]== "<+^+L" :
                self.buttn.left=True
                self.buttn.up=True
                self.buttn.L= not (player.player_buttons.L)
                print("<+^+L")
            elif self.remaining_code[0]== "!<+!^+!L" :
                self.buttn.left=False
                self.buttn.up=False
                self.buttn.L= False  #not (player.player_buttons.Y)
                print("!<+!^+!L")

            elif self.remaining_code[0]== "<+^+Y" :
                self.buttn.left=True
                self.buttn.up=True
                self.buttn.Y= not (player.player_buttons.Y)
                print("<+^+Y")
            elif self.remaining_code[0]== "!<+!^+!Y" :
                self.buttn.left=False
                self.buttn.up=False
                self.buttn.Y= False  #not (player.player_buttons.Y)
                print("!<+!^+!Y")

            elif self.remaining_code[0]== "<+^+R" :
                self.buttn.left=True
                self.buttn.up=True
                self.buttn.R= not (player.player_buttons.R)
                print("<+^+R")
            elif self.remaining_code[0]== "!<+!^+!R" :
                self.buttn.left=False
                self.buttn.up=False
                self.buttn.R= False  #not (player.player_buttons.Y)
                print("!<+!^+!R")

            elif self.remaining_code[0]== "<+^+A" :
                self.buttn.left=True
                self.buttn.up=True
                self.buttn.A= not (player.player_buttons.A)
                print("<+^+A")
            elif self.remaining_code[0]== "!<+!^+!A" :
                self.buttn.left=False
                self.buttn.up=False
                self.buttn.A= False  #not (player.player_buttons.Y)
                print("!<+!^+!A")

            elif self.remaining_code[0]== "<+^+B" :
                self.buttn.left=True
                self.buttn.up=True
                self.buttn.B= not (player.player_buttons.B)
                print("<+^+B")
            elif self.remaining_code[0]== "!<+!^+!B" :
                self.buttn.left=False
                self.buttn.up=False
                self.buttn.B= False  #not (player.player_buttons.Y)
                print("!<+!^+!B")

            elif self.remaining_code[0]== "v+R" :
                self.buttn.down=True
                self.buttn.R= not (player.player_buttons.R)
                print("v+R")
            elif self.remaining_code[0]== "!v+!R" :
                self.buttn.down=False
                self.buttn.R= False  #not (player.player_buttons.Y)
                print("!v+!R")

            else:
                if self.remaining_code[0] =="v" :
                    self.buttn.down=True
                    print ( "down" )
                elif self.remaining_code[0] =="!v":
                    self.buttn.down=False
                    print ( "Not down" )
                elif self.remaining_code[0] =="<" :
                    print ( "left" )
                    self.buttn.left=True
                elif self.remaining_code[0] =="!<" :
                    print ( "Not left" )
                    self.buttn.left=False
                elif self.remaining_code[0] ==">" :
                    print ( "right" )
                    self.buttn.right=True
                elif self.remaining_code[0] =="!>" :
                    print ( "Not right" )
                    self.buttn.right=False

                elif self.remaining_code[0] =="^" :
                    print ( "up" )
                    self.buttn.up=True
                elif self.remaining_code[0] =="!^" :
                    print ( "Not up" )
                    self.buttn.up=False
            self.remaining_code=self.remaining_code[1:]
        return

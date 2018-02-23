import { ScreamProvider } from './../../providers/scream/scream';
import { Component, OnInit } from '@angular/core';
import { NavController, NavParams } from 'ionic-angular';

/**
 * Generated class for the MessagePage page.
 *
 * See https://ionicframework.com/docs/components/#navigation for more info on
 * Ionic pages and navigation.
 */

@Component({
  selector: 'page-message',
  templateUrl: 'message.html',
})
export class MessagePage implements OnInit {

  messages: any;
  newMode: Boolean = true;
  phase1: Boolean = false;
  phase2: Boolean = false;
  phase3: Boolean = false;
  startover: Boolean = false;
  textMessage: any;


  love= '../assets/imgs/love/minion.jpg';
  happy= '../assets/imgs/happy/funny.jpg';
  sadness= ['../assets/imgs/sadness/bean.gif','../assets/imgs/sadness/cat.gif','../assets/imgs/sadness/dance.gif'];
  


  first: string = 'Awesome';
  second: string = 'Screaming';
  third: string = 'Holding it together, actually';

  constructor(public navCtrl: NavController,
    public navParams: NavParams,
    public scream: ScreamProvider) {
  }


  ngOnInit() {
 this.loadOne();

  }


  loadOne() {
    this.messages = [];
    this.startover = false;
    this.newMode = true;

    this.addMessage('Good Evening, Prabh!',null, 1);
    this.addMessage('How are you doing?',null, 1);
  }

  ionViewDidLoad() {
  }

  sendMessage() {
    if(!this.phase1 && !this.phase2 && !this.phase3 && !this.newMode && this.textMessage){
      this.addMessage(this.textMessage, null, 2);
      const message = this.textMessage;
      this.textMessage = null;
     this.apiCall(message);
    }
  }

  addMessage(message,link, type) {
   if(type == 1) {
     this.messages.push({
       message: message,
       text_class: 'left_text',
       pop_class: 'left_one', 
       link: link 
     })

   } else if (type == 2) {
     this.messages.push({
       message: message,
       text_class: 'right_text',
       pop_class: 'right_one',
       link: link
     })
   }

  }


  lets_begin(button_number) {
    if(button_number == 1) {
      this.addMessage(this.first, null,2);
      this.newMode = false;
    } else if (button_number == 2) {
      this.addMessage(this.second, null,2);
      this.newMode = false;
      this.phase2 = true;
      this.addMessage('Hmm, you wanna talk about it?',null, 1);

    } else if(button_number == 3) {
      this.addMessage(this.third,null, 2);
      this.newMode = false;
      
    }


  }


  yesButton() {
    if(this.phase1){

    }
    else if(this.phase2) {
      this.addMessage('Yea, Sure',null, 2);
      this.phase2 = false;

    }
    else if(this.phase3) {

    }

  }
noButton() {
 if(this.phase1){

    }
    else if(this.phase2) {
      this.addMessage('Nah!',null, 2);
      this.phase2 = false;
      this.startover = true;

   this.addMessage('Wanna see some cartoon?', '../assets/imgs/cattap.gif', 1);

    }
    else if(this.phase3) {

    }
}


  apiCall(message) {
  this.scream.getSentiments(message)
  .subscribe((res)=> {
    if(res) {
      console.log(res,'res');
     this.showRes(res.res);
    }
  })

}

  showRes(res) {
    if(res == 'worry') {
      this.addMessage(`We've details of a good psychologist nearby you, I'd like you to contact him once... 
      1. Dr. Deepak Shetty (9686762421) 
       2. Dr. Shweta Mahajan (9876543210)  `, null, 1)
    } else if(res == 'sadness') {
      
      this.addMessage('Cheer up man!', this.sadness[Math.floor((Math.random() * 2) + 0)],1);
    } else if(res == 'love') {
      this.addMessage('Oh! somebody\'s in love...',this.love, 1);
    } else if(res == 'happiness') {
      this.addMessage('that makes me happy too...', this.happy, 1);
      
    } else {
      this.addMessage('Can you try again, i missed it', '../assets/imgs/mark.png', 1);
      
    }

  }



}

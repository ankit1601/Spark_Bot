import { HttpClient } from '@angular/common/http';
import { MessagePage } from './../pages/message/message';
import { SplashPage } from './../pages/splash/splash';
import { ErrorHandler, NgModule } from '@angular/core';
import { IonicApp, IonicErrorHandler, IonicModule } from 'ionic-angular';
import { BrowserModule } from '@angular/platform-browser';
import { SplashScreen } from '@ionic-native/splash-screen';
import { StatusBar } from '@ionic-native/status-bar';
import { HttpModule } from '@angular/http';

import { MyApp } from './app.component';
import { HomePage } from '../pages/home/home';
import { ScreamProvider } from '../providers/scream/scream';

@NgModule({
  declarations: [
    MyApp,
    HomePage,
    SplashPage,
    MessagePage
    
  ],
  imports: [
    BrowserModule,
    HttpModule,
    IonicModule.forRoot(MyApp)
  ],
  bootstrap: [IonicApp],
  entryComponents: [
    MyApp,
    HomePage,
    SplashPage,
    MessagePage
    
  ],
  providers: [
    StatusBar,
    SplashScreen,
    ScreamProvider,
    HttpModule,
    HttpClient,
    {provide: ErrorHandler, useClass: IonicErrorHandler},
    ScreamProvider
  ]
})
export class AppModule {}

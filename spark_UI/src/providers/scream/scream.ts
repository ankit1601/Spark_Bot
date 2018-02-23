import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs/Rx';
import { Http} from '@angular/http';
import 'rxjs/add/operator/map'

/*
  Generated class for the ScreamProvider provider.

  See https://angular.io/guide/dependency-injection for more info on providers
  and Angular DI.
*/
@Injectable()
export class ScreamProvider {

  getUrlForSentiment = 'http://10.201.23.37:5000/get/sent?text=';

  constructor(private http: Http) {
    console.log('Hello ScreamProvider Provider');
  }


  getSentiments(query) {
    return this.http.get(`${this.getUrlForSentiment}"${query}"`)
    .map(res=> res.json());

  }


}

```python
z_q = z_e + (z_q - z_e).detach()
```
- 사용 시
```bash
[ DEVICE: mps ][ N_CPUS: 7 ]
[ 1/50 ][ Train loss: 1.371 ][ Val loss: 1.292 | 1.292 ]                   
[ 2/50 ][ Train loss: 1.260 ][ Val loss: 1.232 | 1.232 ]                   
[ 3/50 ][ Train loss: 1.184 ][ Val loss: 1.180 | 1.180 ]                   
[ 4/50 ][ Train loss: 1.111 ][ Val loss: 1.076 | 1.076 ]                   
[ 5/50 ][ Train loss: 1.034 ][ Val loss: 0.990 | 0.990 ]                   
[ 6/50 ][ Train loss: 0.953 ][ Val loss: 0.941 | 0.941 ]                   
[ 7/50 ][ Train loss: 0.869 ][ Val loss: 0.832 | 0.832 ]                   
[ 8/50 ][ Train loss: 0.784 ][ Val loss: 0.763 | 0.763 ]                   
[ 9/50 ][ Train loss: 0.700 ][ Val loss: 0.665 | 0.665 ]                   
[ 10/50 ][ Train loss: 0.619 ][ Val loss: 0.605 | 0.605 ]                  
[ 11/50 ][ Train loss: 0.547 ][ Val loss: 0.529 | 0.529 ]                  
[ 12/50 ][ Train loss: 0.484 ][ Val loss: 0.454 | 0.454 ]                  
[ 13/50 ][ Train loss: 0.430 ][ Val loss: 0.410 | 0.410 ]                  
[ 14/50 ][ Train loss: 0.384 ][ Val loss: 0.358 | 0.358 ]                  
[ 15/50 ][ Train loss: 0.346 ][ Val loss: 0.332 | 0.332 ]                  
[ 16/50 ][ Train loss: 0.315 ][ Val loss: 0.305 | 0.305 ]                  
[ 17/50 ][ Train loss: 0.293 ][ Val loss: 0.294 | 0.294 ]                  
[ 18/50 ][ Train loss: 0.275 ][ Val loss: 0.267 | 0.267 ]                  
[ 19/50 ][ Train loss: 0.260 ][ Val loss: 0.254 | 0.254 ]                  
[ 20/50 ][ Train loss: 0.247 ][ Val loss: 0.244 | 0.244 ]                  
[ 21/50 ][ Train loss: 0.237 ][ Val loss: 0.237 | 0.237 ]                  
[ 22/50 ][ Train loss: 0.228 ][ Val loss: 0.242 | 0.237 ]                  
[ 23/50 ][ Train loss: 0.221 ][ Val loss: 0.216 | 0.216 ]                  
[ 24/50 ][ Train loss: 0.217 ][ Val loss: 0.217 | 0.216 ]                  
[ 25/50 ][ Train loss: 0.209 ][ Val loss: 0.209 | 0.209 ]                  
[ 26/50 ][ Train loss: 0.204 ][ Val loss: 0.200 | 0.200 ]                  
[ 27/50 ][ Train loss: 0.200 ][ Val loss: 0.196 | 0.196 ]                  
[ 28/50 ][ Train loss: 0.197 ][ Val loss: 0.191 | 0.191 ]                  
[ 29/50 ][ Train loss: 0.195 ][ Val loss: 0.192 | 0.191 ]                  
[ 30/50 ][ Train loss: 0.192 ][ Val loss: 0.209 | 0.191 ]                  
[ 31/50 ][ Train loss: 0.189 ][ Val loss: 0.188 | 0.188 ]                  
[ 32/50 ][ Train loss: 0.186 ][ Val loss: 0.198 | 0.188 ]                  
```
- 미사용 시
```bash
[ DEVICE: mps ][ N_CPUS: 7 ]
[ 1/50 ][ Train loss: 1.378 ][ Val loss: 1.265 | Best: 1.265 ]             
[ 2/50 ][ Train loss: 1.132 ][ Val loss: 1.042 | Best: 1.042 ]             
[ 3/50 ][ Train loss: 0.918 ][ Val loss: 0.814 | Best: 0.814 ]             
[ 4/50 ][ Train loss: 0.718 ][ Val loss: 0.641 | Best: 0.641 ]             
[ 5/50 ][ Train loss: 0.565 ][ Val loss: 0.498 | Best: 0.498 ]
```

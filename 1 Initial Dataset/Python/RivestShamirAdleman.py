import random
import math

def Rabin_Miller_Test(d, n):
    a = random.randrange(2, n-2)
    x = pow(a, d, n) # pow(base, exp, mod)
    if (x == 1 or x == n-1):
        return True
    while (d != n - 1):
        x = (x * x) % n
        d *= 2
 
        if (x == 1):
            return False
        if (x == n - 1):
            return True

    return False

def isPrime (n, k):
    if (n <= 1 or n == 4): # quick dummy check
        return False
    if (n <= 3):
        return True

    d = n - 1
    while (d % 2 == 0):
        d //= 2
 
    for i in range(k):
        if (Rabin_Miller_Test(d, n) == False):
            return False
 
    return True

# rounds = math.log(n)/math.log(4) - number of needed rounds
P = random.randrange(300000000000000000, 10000000000000000000)
P = P | 1
while isPrime (P, 8) == False:
    P = P | 1
    P = P + 2

Q = random.randrange(300000000000000000, 100000000000000000000)
Q = Q | 1
while isPrime (Q, 8) == False:
    Q = Q | 1
    Q = Q + 2

print("P is " + str(P))
print("Q is " + str(Q))

N = (P * Q)
print("Modulo is " + str(N))

phi = (P-1)*(Q-1)
print("φ is " + str(phi))

while True:
    e = random.randrange(2, phi-1)
    e = e | 1
    while isPrime (e, 8) == False:
        e = e | 1
        e = e + 2
    g = math.gcd(e, phi)
    d = pow(e, -1, phi)
    check = d*e % phi
    print("Check is " + str(check))
    print("GCD is " + str(g))
    if g == 1 and check == 1:
        break

print("e is " + str(e))
print("d is " + str(d))

def defineKeypair (e, d, N):
    return ((e, N), (d, N))

def encrypt(publicKey, message):
    # Unpack the key 
    e, n = publicKey
    # Convert each letter of plaintext to ascii using plain^e mod N
    c = [pow(ord(char), e, n) for char in message]
    return c

def decrypt(privateKey, message):
    # Unpack the key 
    d, n = privateKey
    # Generate the plaintext using cipher^d mod N
    p = [chr(pow(char, d, n)) for char in message]
    # Return the array 
    return ''.join(p)

publicKey , privateKey = defineKeypair (e, d, N)

# Entering the message to be encrypted
M = input('Enter your message:  ')
print('Public Key [e,n] = ', publicKey)

# Encryption
C = encrypt(publicKey, M)
# Decryption
M = decrypt(privateKey, C)

print('Cipher text is:  ', C)
print('Private Key [d,n] = ', privateKey)
print('Plain text after decryption is: ', M)
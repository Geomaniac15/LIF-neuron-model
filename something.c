void do_something() {
    // Code with O(1) time complexity
}

void main() {
    int n = 10; // Example input size
    
    for (int i=0; i<10; i++) {
        for(int j=1000; j>=n; j/2) {
            do_something();
        }
    }
}
#include <stdlib.h>
#include <stdio.h>

struct node {
   int val;
   struct node* l;
   struct node* r;
} nodeT;

#define MAXLEVEL 5

struct node* build(int level) {

   if (level < MAXLEVEL) {
      struct node* p = (struct node*) malloc(sizeof(nodeT));
      p->val = level;
      p->l = build(level+1);
      p->r = build(level+1);
      return p;
   } else {
      return NULL;
  }
}

void traverse(struct node* p) {
   if (p == NULL) return;
   if (p->l == NULL) return;
   else traverse(p->l);
   if (p->r == NULL) return;
   else traverse(p->r);
   printf("%d\n", p->val);
}

int main( ) {

   struct node* h = build(0);
   traverse(h);


}

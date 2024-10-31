 u2=load('12001');
max(max(u2))
image(160+255*u2/max(max(u2)));
colormap gray;axis off;axis image;axis xy;axis tight;axis equal;
%saveas(gcf,'1300.jpg','tiff')